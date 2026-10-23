"""Pattern drift scoring — the evaluation framework.

**Ground truth** is a reference role mapping of a codebase; the **recovery result**
is what a recovery produced for the same codebase and pattern. Drift is how far the
recovery result strayed from the ground truth.

Each level of the framework is a :class:`~.base.PatternMetric` subclass, so
alternative formulas for the same level coexist and are selected here:

>>> DEFAULT_METRICS = [RoleAgreementDetection(), PathLengthGrounding(), ...]

Changing a formula means changing one entry in that list. Nothing else in the
codebase names a concrete metric class.

Consumers do not run the levels themselves: :func:`~.evaluation.evaluate` is the
one entry point, taking an :class:`~.evaluation.EvaluationRequest` and returning
one result per level. Every consumer goes through it, so no two can score a level
differently — see ``evaluation.py``.
"""

from __future__ import annotations

import logging
import re

from .base import (
    DETECTED_STATUSES,
    LEVEL1_CLASSES,
    MetricResult,
    PatternMetric,
    ScoringContext,
    is_detected,
    level1_class,
)
from .containment.containment_tree import ContainmentTree
from .level1_detection import MacroF1RoleDetection, RoleAgreementDetection
from .level2_grounding import PathLengthGrounding
from .level3_order import InteractionOrder, level3_score

logger = logging.getLogger(__name__)

#: The active formula for each level, in reporting order, one entry per level.
#: Level 3 is interaction overlap × the order of the shared interactions; see
#: level3_order.py.
#: Level 1 alternative available: MacroF1RoleDetection (the metrics summary's
#: macro-F1 formulation) — see level1_detection.py for why agreement-rate is the
#: default on a single codebase.
DEFAULT_METRICS: list[PatternMetric] = [
    RoleAgreementDetection(),
    PathLengthGrounding(),
    InteractionOrder(),
]


def score_pair(
    ctx: ScoringContext, metrics: list[PatternMetric] | None = None
) -> list[MetricResult]:
    """Run every metric over one (codebase, pattern) pair.

    A metric that fails outright is reported as unavailable rather than taking the
    whole score down with it — a broken Level 2 must not hide a working Level 1.
    """
    results: list[MetricResult] = []
    for metric in metrics if metrics is not None else DEFAULT_METRICS:
        try:
            results.append(metric.score(ctx))
        except Exception:  # pragma: no cover - defensive
            logger.exception("%s raised; reported as unavailable", type(metric).__name__)
            results.append(
                MetricResult(
                    key=metric.key,
                    level=metric.level,
                    formula=metric.formula,
                    available=False,
                    headline_label=metric.headline_label,
                )
            )
    return results


def overall_drift(results: list[MetricResult]) -> float | None:
    """Composite drift: 1 - mean(score) over the *levels* that produced one.

    Averaged per level first, then across levels, so a level reporting several
    sub-metrics (keys sharing a level prefix, e.g. ``level1a``/``level1b``) would
    not thereby weigh more than one reporting a single number.

    Unavailable levels are excluded rather than counted as zero — a level that
    cannot be measured is not evidence of drift. Returns None when nothing scored.
    """
    by_level: dict[str, list[float]] = {}
    for r in results:
        if r.available and r.score is not None:
            # A trailing letter names a sub-metric: "level1a" belongs to "level1".
            by_level.setdefault(re.sub(r"[a-z]+$", "", r.key) or r.key, []).append(r.score)
    if not by_level:
        return None
    levels = [sum(s) / len(s) for s in by_level.values()]
    return 1.0 - sum(levels) / len(levels)


def build_role_vocabulary(
    pattern: dict | None, ground_truth: dict[str, dict], recovery_result: dict[str, dict]
) -> list[str]:
    """Resolve Level 1's fixed label space.

    Prefers the reference pattern's declared roles (``security_patterns.roles``).
    Falls back to the union of roles named by either side, which is weaker: a role
    both sides overlooked becomes invisible instead of a shared false negative.
    """
    if pattern:
        names = [
            (r.get("name") or "").strip()
            for r in pattern.get("roles", [])
            if (r.get("name") or "").strip()
        ]
        if names:
            # Roles the recovery result cites but the pattern does not are still
            # scored — dropping them would hide hallucinated roles entirely. Roles
            # only the ground truth cites are not: the recovery is asked for the
            # pattern's roles, so it can never agree or disagree on them.
            extras = sorted(set(recovery_result) - set(names))
            return names + extras

    return sorted(set(ground_truth) | set(recovery_result))


# Imported last, not with the block at the top: ``evaluation`` is the pipeline
# *over* this registry and reads DEFAULT_METRICS/score_pair/overall_drift/
# build_role_vocabulary from here, so the names above must exist before it loads.
from .evaluation import (  # noqa: E402
    SCOPE_ALL,
    SCOPE_REQUIRED,
    EvaluationReport,
    EvaluationRequest,
    RoleSelection,
    ScoredInputs,
    evaluate,
    index_roles,
    resolve_roles,
    scored_behaviour,
    scored_mappings,
)

__all__ = [
    "DEFAULT_METRICS",
    "DETECTED_STATUSES",
    "LEVEL1_CLASSES",
    "SCOPE_ALL",
    "SCOPE_REQUIRED",
    "ContainmentTree",
    "EvaluationReport",
    "EvaluationRequest",
    "InteractionOrder",
    "MacroF1RoleDetection",
    "RoleAgreementDetection",
    "MetricResult",
    "PathLengthGrounding",
    "PatternMetric",
    "RoleSelection",
    "ScoredInputs",
    "ScoringContext",
    "build_role_vocabulary",
    "evaluate",
    "index_roles",
    "is_detected",
    "level1_class",
    "level3_score",
    "overall_drift",
    "resolve_roles",
    "score_pair",
    "scored_behaviour",
    "scored_mappings",
]
