"""The one call that runs the evaluation framework — used by every consumer.

Callers differ only in *where the inputs come from*. Everything after that —
indexing the rows by role, resolving the label space, restricting both sides to
it, building the containment tree, running each level, compositing the drift —
lives here, once. A caller assembles an :class:`EvaluationRequest` out of whatever
it has and calls :func:`evaluate`; what comes back is an :class:`EvaluationReport`
holding one :class:`~.base.MetricResult` per level, in registry order, plus the
provenance a caller displays. No caller touches ``score_pair``, ``ScoringContext``
or a metric class directly, so no two can score a level differently.

What stays caller-side is only input acquisition: reading rows, parsing the repo
into ``struct``, and binding the Level 3 detector to the repo.

Row shape is the recovery ``role_mappings`` schema on both sides. A caller that
validates rows passes its validator as ``normalise_row``; one that trusts its
input passes nothing, and every field the metrics read passes through untouched
either way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from secpat_interaction_order import DetectionResult

from . import DEFAULT_METRICS, build_role_vocabulary, overall_drift, score_pair
from .base import MetricResult, PatternMetric, ScoringContext
from .containment.containment_tree import ContainmentTree

#: ``RoleSelection.scope`` values — *which* of the pattern's roles are scored.
SCOPE_ALL = "all"
SCOPE_REQUIRED = "required"


@dataclass(frozen=True)
class RoleSelection:
    """The label space one evaluation scores over."""

    #: The roles scored, in reporting order.
    names: list[str]
    #: ``"all"`` or ``"required"`` — see :data:`SCOPE_ALL`.
    scope: str = SCOPE_ALL
    #: Roles the pattern defines but this run deliberately does not score.
    excluded: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """One phrase for a report header, e.g. "8 roles (required only)"."""
        return "required only" if self.scope == SCOPE_REQUIRED else "all roles"


def resolve_roles(
    pattern: dict | None,
    ground_truth: dict[str, dict],
    recovery_result: dict[str, dict],
) -> RoleSelection:
    """Resolve the label space Level 1 scores over.

    **A role the pattern flags ``required: false`` is excluded, always.** The
    ``Subject`` is the browser and the ``System`` is the business logic being
    protected; every codebase has both, neither is what the pattern prescribes,
    and an optional role a codebase legitimately does not implement is not drift.

    There is no caller switch for this: a switch would let the same scoring code
    return different Level 1 and Level 3 numbers for the same two sides depending
    on which caller asked, which is exactly what one shared package exists to
    prevent.

    **Only an explicit ``false`` excludes.** A role with no flag — or ``null``,
    which is what a model declaring the field optional writes on dump — is
    required. The flag was once read as all-or-nothing: the moment one role
    carried it, every role *without* it was dropped too, so marking just
    ``Subject`` optional by hand excluded the whole pattern and scored nothing.
    Defaulting to required makes a partly flagged pattern mean what it says, and
    an unflagged one — e.g. a document predating the flag — is scored in full
    without needing a case of its own.

    Every role the pattern declares and does not mark optional is scored, plus
    any role the recovery result names that the pattern does not. Dropping those
    extras would hide a hallucinated role entirely, so they are scored and
    mismatch. A role only the ground truth names is not scored (secpat-scoring
    0.22.0): the recovery is asked for the pattern's roles and never names it, so
    it would only ever add a free agreement (ground truth `missing`) or an
    unavoidable mismatch (anything else) — annotations copied from a recovery
    that used the catalogue's longer role list carry such rows.

    Falls back to the union of both sides' roles when there is no usable pattern,
    which is weaker: a role both sides overlooked becomes invisible instead of a
    shared false negative.
    """
    roles = (pattern or {}).get("roles", [])
    declared = [(r.get("name") or "").strip() for r in roles]
    declared = [n for n in declared if n]
    optional = [
        name for r in roles
        if r.get("required") is False and (name := (r.get("name") or "").strip())
    ]

    if declared and optional:
        # A role the recovery names but the pattern does not declare is a
        # hallucination, and it is still scored: dropping it would hide it
        # entirely. One only the ground truth names is not. Only a *declared*
        # role can be excluded, and only by its own `required: false`.
        extras = sorted(set(recovery_result) - set(declared))
        return RoleSelection(
            names=[n for n in declared if n not in set(optional)] + extras,
            scope=SCOPE_REQUIRED,
            excluded=optional,
        )

    return RoleSelection(
        names=build_role_vocabulary(pattern, ground_truth, recovery_result),
        scope=SCOPE_ALL,
    )


def restrict(side: dict[str, dict], roles: RoleSelection) -> tuple[dict[str, dict], list[str]]:
    """One side's rows limited to the scored label space, plus the roles dropped."""
    scored = set(roles.names)
    return (
        {role: row for role, row in side.items() if role in scored},
        sorted(set(side) - scored),
    )


def scored_behaviour(pattern: dict | None, excluded: list[str]) -> list[dict]:
    """The pattern's walkthrough minus every step to or from an excluded role.

    A step whose endpoint is not scored describes an interaction nothing is
    derived for, so it is removed from the pattern for the whole evaluation: the
    walk does not take its entry point or tie-breaks from it, and Level 3 does not
    count it among the prescribed steps it reports coverage of. Steps keep their
    own numbers — the walk orders by them, and a gap is harmless.
    """
    out = set(excluded)
    return [
        step for step in (pattern or {}).get("behaviour") or []
        if (step.get("from") or "").strip() not in out
        and (step.get("to") or "").strip() not in out
    ]


class ScoredInputs(NamedTuple):
    """One side's inputs to a walk, limited to what Level 3 scores."""

    #: The mapping to walk — ``required: false`` roles removed.
    mappings: dict[str, dict]
    #: The roles removed from ``mappings``.
    excluded: list[str]
    #: The pattern's walkthrough without the steps touching an excluded role.
    behaviour: list[dict]


def scored_mappings(pattern: dict | None, mappings: dict[str, dict]) -> ScoredInputs:
    """One mapping and walkthrough limited to what Level 3 scores, for a walk outside :func:`evaluate`.

    :func:`evaluate` restricts both sides and the walkthrough before any metric
    sees them, so the walk behind a score never reaches a ``required: false``
    role or a step that involves one. A one-sided order derived anywhere else —
    a recovery run's stage 3, say — must be restricted
    the same way, or the pattern's opening speaker (usually ``Subject``) becomes
    the entry of an order no score uses.
    """
    # Passed as the recovery side so a role the pattern does not declare is still
    # walked, whichever side these rows come from.
    roles = resolve_roles(pattern, {}, mappings)
    walked, excluded = restrict(mappings, roles)
    return ScoredInputs(walked, excluded, scored_behaviour(pattern, roles.excluded))


def index_roles(
    rows: list[dict], normalise: Callable[[dict], dict] | None = None
) -> dict[str, dict]:
    """Index role rows by pattern role.

    A row with no ``patternRole`` cannot be paired against anything, so it is
    dropped. On a duplicate role the first row wins, since one row per role is the
    intent. ``normalise`` — a caller's row validation — is applied per row when
    given.
    """
    indexed: dict[str, dict] = {}
    for raw in rows:
        row = normalise(raw) if normalise else raw
        role = (row.get("patternRole") or "").strip()
        if role and role not in indexed:
            indexed[role] = row
    return indexed


@dataclass(frozen=True)
class EvaluationRequest:
    """One (codebase, pattern) pair to score, however its caller obtained it.

    Rows go in raw — indexing, label-space resolution and restriction all happen
    inside :func:`evaluate`, so two callers cannot disagree about them.
    """

    #: The reference mapping, as ``role_mappings`` rows.
    ground_truth_rows: list[dict]
    #: What the recovery pipeline produced, as ``role_mappings`` rows.
    recovery_result_rows: list[dict]
    #: The reference pattern document — supplies both the role vocabulary and
    #: Level 3's behaviour walkthrough, which must come from one revision.
    pattern: dict | None = None
    #: file path -> symbol summary. Level 2 consumes it as a containment tree,
    #: Level 3 needs the raw line ranges.
    struct: dict[str, dict] | None = None
    #: Derives one side's interaction order — see ``ScoringContext.detect_order``.
    #: ``None`` means Level 3 reports itself unavailable.
    detect_order: Callable[[dict[str, dict], list[dict]], DetectionResult] | None = None
    #: Per-row validation, applied to both sides.
    normalise_row: Callable[[dict], dict] | None = None
    #: Formula per level. Defaults to the registry in ``secpat_scoring/__init__.py``.
    metrics: list[PatternMetric] | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """One :class:`~.base.MetricResult` per level, plus how the run was set up."""

    #: One per registered metric, in registry order — level 1, 2, 3a, 3b.
    metrics: list[MetricResult]
    #: Composite: ``1 - mean(score)`` over the levels that produced one.
    overall_drift: float | None
    roles: RoleSelection
    #: True when a containment tree was built — i.e. Level 2 had its input.
    grounding_available: bool
    #: Rows naming a role outside the scored label space, dropped before scoring.
    dropped_ground_truth_rows: list[str] = field(default_factory=list)
    dropped_recovery_result_rows: list[str] = field(default_factory=list)
    #: How many rows each side contributed after restriction.
    ground_truth_roles: int = 0
    recovery_result_roles: int = 0

    def level(self, key: str) -> MetricResult | None:
        """One metric's result by key ("level1", "level2", "level3")."""
        return next((m for m in self.metrics if m.key == key), None)

    def to_dict(self) -> dict:
        """The camelCase shape callers serialise and render."""
        return {
            "roleVocabulary": list(self.roles.names),
            "roleScope": self.roles.scope,
            "excludedRoles": list(self.roles.excluded),
            "droppedGroundTruthRows": list(self.dropped_ground_truth_rows),
            "droppedRecoveryResultRows": list(self.dropped_recovery_result_rows),
            "groundTruthRoles": self.ground_truth_roles,
            "recoveryResultRoles": self.recovery_result_roles,
            "groundingAvailable": self.grounding_available,
            "overallDrift": (
                None if self.overall_drift is None else round(self.overall_drift, 4)
            ),
            "metrics": [m.to_dict() for m in self.metrics],
        }


def evaluate(request: EvaluationRequest) -> EvaluationReport:
    """Run every level of the framework over one (codebase, pattern) pair.

    The whole pipeline, in the order both callers used to spell out for
    themselves: index both sides by role, resolve the label space, restrict both
    sides to it, build the containment tree, score each level, composite the drift.

    Never raises for data reasons — a level lacking its input reports itself
    unavailable and is excluded from the composite, rather than taking the other
    levels down with it.
    """
    ground_truth = index_roles(request.ground_truth_rows, request.normalise_row)
    recovery_result = index_roles(request.recovery_result_rows, request.normalise_row)

    roles = resolve_roles(request.pattern, ground_truth, recovery_result)

    # Restrict both sides to the scored label space *before* any metric sees them,
    # so a role left out is absent rather than counted as a shared negative. A
    # no-op unless the pattern narrowed it with `required: false`: the default
    # vocabulary already absorbs every role either side names.
    ground_truth, dropped_gt = restrict(ground_truth, roles)
    recovery_result, dropped_rr = restrict(recovery_result, roles)

    tree = ContainmentTree(request.struct) if request.struct else None

    results = score_pair(
        ScoringContext(
            ground_truth=ground_truth,
            recovery_result=recovery_result,
            role_vocabulary=roles.names,
            tree=tree,
            behaviour=scored_behaviour(request.pattern, roles.excluded),
            struct=request.struct,
            detect_order=request.detect_order,
        ),
        request.metrics if request.metrics is not None else DEFAULT_METRICS,
    )

    return EvaluationReport(
        metrics=results,
        overall_drift=overall_drift(results),
        roles=roles,
        grounding_available=tree is not None,
        dropped_ground_truth_rows=dropped_gt,
        dropped_recovery_result_rows=dropped_rr,
        ground_truth_roles=len(ground_truth),
        recovery_result_roles=len(recovery_result),
    )
