"""Level 3 — Interaction Order, one number.

    "*Given both endpoint roles of a prescribed interaction were correctly
    grounded*, does the recovered order between interactions match the reference
    pattern's prescribed order?"

Two sides can disagree about **which interactions exist at all**, and, among the
interactions they both found, about **what order they run in**. Level 3 scores
both, as a product of two factors over the same pair of derived orderings:

    level3 = overlap × order

* **overlap** — Jaccard over the two interaction sets, ``|shared| / |union|``.
* **order** — Kendall's Tau over **only the shared interactions**, in each side's
  own relative order (Fagin, Kumar & Sivakumar's induced sub-order, their p = 0),
  reported as the concordant fraction in [0, 1]. Order needs two things to order,
  so the factor is fixed where there are fewer: **1** when exactly one interaction
  is shared (nothing to get wrong), **0** when none is (nothing found in common —
  and overlap is already 0 then, so the product is 0 either way).

**Fewer than two roles both sides detected scores 0.** No interaction can be
formed between agreed roles then, so the recovery shares none of the ground
truth's — the same outcome as "no interaction in common", and scored the same
way. The miss is charged here as well as in Level 1, deliberately: a recovery
that found too little of the pattern to say anything about its order has not
recovered the order. It stays *unavailable* only when the ground truth itself has
fewer than two detected roles, because then there is no ground-truth interaction
for any recovery to match.

**No interaction on either side scores 0 as well** (since 0.20.0): with two or more
agreed roles, neither walk connecting any two of them means the recovery shares no
interaction with the ground truth among the roles it found — the ground truth's
interactions all run through a role the recovery missed. Scored like any other
"nothing in common", overlap 0 and order 0.

A product rather than a mean because neither factor can stand in for the other:
a perfectly ordered pair of interactions out of ten found scores low on overlap,
and a complete set in reverse order scores low on order, and either failure alone
should pull the level down. The induced sub-order is safe to use for the order
factor precisely because overlap charges for the interactions it leaves out.

Both factors run over the same derived orderings and the same conditioning — see
:func:`level3_inputs`, which derives once and memoises on the context — and both
are in ``detail`` beside the product.

The conditioning is literal: **only roles both sides detected take part.**
Level 1 already scores whether a role was found at all and Level 2 whether it was
grounded in the right place. A role only one side detected is *their* finding to
report, and letting it into Level 3 charges the same mistake twice — worse, it
distorts the very thing Level 3 exists to measure. So the role set is intersected
first (see :func:`agreed_roles`) and both orderings are derived over that set
alone. A role dropped this way is not a hole in the walk: it stops being an anchor
and becomes an ordinary function, so the walk passes *through* it exactly as it
passes through any other non-role helper, and both sides walk the same roles.

**What the recovered order is compared against — the ground truth.** The
framework phrases Level 3 against the reference pattern's prescribed order, but
every other level measures the recovery result against the ground truth, and drift is defined as the distance between
those two. So Level 3 does the same: an ordering is derived **independently from
each side's role mappings by the same detector**, and the score is how far the
recovery result's ordering strayed from the ground truth's.
``SecurityPattern.behaviour`` is still used, and used for more than before: it
**names the entry role** each side's walk begins at (step 1's ``from``, falling
through to the next step's speaker when that role was not recovered), it breaks
ties between call-graph roots, and it supplies the coverage numbers in ``detail``.
What it is *not* is the thing being compared against — the pattern says where the
interaction starts, the reviewer's ground truth says what the correct order is.

Where the orderings come from
─────────────────────────────

Neither side ever recorded an interaction order; ``role_mappings`` says which
entity fills each role, not when it runs. The order is inferred statically from
the code by ``secpat_interaction_order`` — read that package's docstring for the
algorithm. The detector is given the agreed role set, not the raw mappings; it
knows nothing about the conditioning and does not need to.

Alternative order formulas (Normalised LCS, Normalised Edit Distance — the
metrics summary lists both, and Anika & Miah (2025) argue for reporting several)
belong in sibling classes here sharing the same ``key``. The sequence *detection*
is shared; only the comparison is formula-specific, which is why the formula lives
on the class rather than in ``scoring_functions.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from secpat_interaction_order import DetectionResult

from .base import MetricResult, PatternMetric, ScoringContext, is_detected

#: Where :func:`level3_inputs` memoises its derivation on ``ScoringContext.cache``.
_CACHE_KEY = "level3-orderings"


def agreed_roles(ctx: ScoringContext) -> set[str]:
    """Roles both sides detected — the only ones Level 3 may consider.

    "Detected" is Level 1's own test (:func:`base.is_detected`), so agreement here
    means agreement there: present, insecure and external all count, and a
    role either side left `missing` does not. Note that the two sides need *not*
    agree on where the role lives — that a role was grounded in different places is
    Level 2's measurement, and Level 3 has a legitimate interest in it, because
    grounding a role somewhere else is exactly what reorders the interactions.

    **Both sides must also name an entity.** Level 3 is derived by locating each
    role in the graph, and the walk locates by name: a row citing only a line
    range names nothing, so it can anchor nothing and contributes no interaction
    it could legitimately have. Worse than nothing, in fact — an entity-less row
    used to be filed under the empty string and matched every nameless reach in
    its file, inventing edges out of `data` and dependency-injection sites that
    had no relation to it. Requiring the name here is what makes the derivation
    honest; Level 2 is where a difference in *where* a role was grounded is
    measured, and it keeps scoring those rows either way.

    Shared by every Level 3 formula: the conditioning belongs to the level, not to
    any one formula.
    """
    return {
        role
        for role in set(ctx.ground_truth) | set(ctx.recovery_result)
        if _usable(ctx.ground_truth.get(role)) and _usable(ctx.recovery_result.get(role))
    }


def _usable(row: dict | None) -> bool:
    """A row Level 3 can anchor: detected by Level 1's test, and naming an entity."""
    return is_detected(row) and bool(((row or {}).get("entity") or "").strip())


@dataclass(frozen=True)
class Level3Inputs:
    """One derivation of both sides' orderings, shared by every Level 3 formula."""

    agreed: set[str]
    #: Roles one side detected and the other did not — left out of both walks.
    excluded: list[str]
    ground_truth: DetectionResult
    recovery_result: DetectionResult

    @property
    def gt_tokens(self) -> list[str]:
        return self.ground_truth.tokens

    @property
    def rr_tokens(self) -> list[str]:
        return self.recovery_result.tokens

    @property
    def shared(self) -> set[str]:
        return set(self.gt_tokens) & set(self.rr_tokens)

    @property
    def union(self) -> set[str]:
        return set(self.gt_tokens) | set(self.rr_tokens)


def level3_inputs(ctx: ScoringContext) -> Level3Inputs | None:
    """Condition the roles and derive both orderings — once per scoring run.

    Returns the derivation, or None when the level cannot be measured at all (no
    way to derive an order, or fewer than two roles both sides detected).

    Deriving an ordering means walking the call graph, which is the expensive part
    of Level 3, so the result is memoised on ``ctx.cache`` and any formula scored
    over the same context reuses the walk.
    """
    if _CACHE_KEY in ctx.cache:
        return ctx.cache[_CACHE_KEY]

    result = _detect_inputs(ctx)
    ctx.cache[_CACHE_KEY] = result
    return result


def _detect_inputs(ctx: ScoringContext) -> Level3Inputs | None:
    if ctx.detect_order is None:
        return None

    # An interaction needs two endpoint roles both sides detected.
    agreed = agreed_roles(ctx)
    if len(agreed) < 2:
        return None

    excluded = sorted((set(ctx.ground_truth) | set(ctx.recovery_result)) - agreed)
    return Level3Inputs(
        agreed=agreed,
        excluded=excluded,
        ground_truth=_detect(ctx, _conditioned(ctx.ground_truth, agreed)),
        recovery_result=_detect(ctx, _conditioned(ctx.recovery_result, agreed)),
    )


def level3_score(gt_tokens: list[str], rr_tokens: list[str]) -> tuple[float, dict] | None:
    """The Level 3 formula on two orderings: ``overlap × order``.

    ``overlap`` is ``|shared| / |union|``; ``order`` is the concordant fraction of
    the shared interactions' pairs (Kendall's Tau over the induced sub-order), or
    1.0 when exactly one interaction is shared and 0.0 when none is. Returns
    ``None`` when neither side found any interaction — there is nothing to compare.

    A function of the two token lists alone, so a stored result that kept both
    sides' ranked interactions can be rescored without walking anything again.
    """
    shared = set(gt_tokens) & set(rr_tokens)
    union = set(gt_tokens) | set(rr_tokens)
    if not union:
        return None

    overlap = len(shared) / len(union)
    if len(shared) >= 2:
        order, pairs = _kendall_tau(gt_tokens, rr_tokens)
    else:
        order = 1.0 if shared else 0.0
        pairs = {
            "rankedInteractions": len(shared),
            "totalPairs": 0,
            "concordantPairs": 0,
            "discordantPairs": 0,
            "kendallTau": None,
        }
    return overlap * order, {
        "interactionOverlap": round(overlap, 4),
        "orderAgreement": round(order, 4),
        **pairs,
    }


def _kendall_tau(seq_a: list[str], seq_b: list[str]) -> tuple[float, dict]:
    """Kendall's Tau over the two sequences restricted to their shared items.

    Each side keeps its own relative order over the intersection; every unordered
    pair of shared interactions is concordant when both sides order it the same
    way and discordant when they order it oppositely. Interactions one side never
    found take part in no pair at all — that is the induced sub-order, and the
    overlap factor is what charges for them.

    Returns the agreement in [0, 1] (the concordant fraction) and the pair
    bookkeeping behind it, including the raw ``τ`` in [−1, 1].
    """
    shared = set(seq_a) & set(seq_b)
    rank_a = {t: i for i, t in enumerate(seq_a) if t in shared}
    rank_b = {t: i for i, t in enumerate(seq_b) if t in shared}

    discordant = 0
    total_pairs = 0
    for i, j in combinations(sorted(shared), 2):
        total_pairs += 1
        if (rank_a[i] < rank_a[j]) != (rank_b[i] < rank_b[j]):
            discordant += 1

    concordant = total_pairs - discordant
    score = concordant / total_pairs if total_pairs else 1.0
    breakdown = {
        "rankedInteractions": len(shared),
        "totalPairs": total_pairs,
        "concordantPairs": concordant,
        "discordantPairs": discordant,
        #: The correlation itself, for readers who want τ rather than the
        #: package's [0, 1] agreement convention. score == (tau + 1) / 2.
        "kendallTau": round((concordant - discordant) / total_pairs, 4) if total_pairs else 1.0,
    }
    return max(0.0, min(1.0, score)), breakdown


class InteractionOrder(PatternMetric):
    """Level 3 — interaction overlap × the order of the shared interactions.

    See the module docstring for the formula and :func:`level3_score` for the
    arithmetic. Reported as an agreement in [0, 1] like every other metric in this
    package; both factors and the pair counts behind the order factor are in
    ``detail``.
    """

    key = "level3"
    level = "Level 3 — Interaction Order"
    formula = "jaccard × kendall-tau (shared interactions)"
    headline_label = "Interaction agreement"

    def score(self, ctx: ScoringContext) -> MetricResult:
        too_few = self._too_few_agreed(ctx)
        if too_few is not None:
            return too_few

        inputs = level3_inputs(ctx)
        if inputs is None:
            return self._unavailable()

        if not (inputs.ground_truth.available and inputs.recovery_result.available):
            # The walk itself failed (no answer, no profile, schema mismatch): an
            # empty ordering here is not evidence of anything, so it must not score.
            return self._unavailable()

        scored = level3_score(inputs.gt_tokens, inputs.rr_tokens)
        if scored is None:
            # Neither walk connected two agreed roles: no interaction is shared.
            scored = (0.0, {
                "interactionOverlap": 0.0, "orderAgreement": 0.0, "rankedInteractions": 0,
                "totalPairs": 0, "concordantPairs": 0, "discordantPairs": 0, "kendallTau": None,
            })
        score, breakdown = scored
        union = inputs.union

        return self._result(
            score=score,
            detail={
                **self._derivation_detail(inputs, ctx),
                "onlyGroundTruth": len(set(inputs.gt_tokens) - inputs.shared),
                "onlyRecoveryResult": len(set(inputs.rr_tokens) - inputs.shared),
                **breakdown,
                "prescribedActionSteps": self._prescribed_steps(ctx),
                "prescribedStepsCovered": self._covered_steps(ctx, union),
            },
            per_role=self._rows(
                inputs.gt_tokens, inputs.rr_tokens, inputs.ground_truth, inputs.recovery_result
            ),
        )

    def _too_few_agreed(self, ctx: ScoringContext) -> MetricResult | None:
        """0 when fewer than two roles were detected by both sides — see the module
        docstring. None when the case does not apply: enough agreed roles, no way to
        derive an order at all, or a ground truth with fewer than two usable roles
        of its own (nothing a recovery could match, so still unavailable)."""
        if ctx.detect_order is None:
            return None
        agreed = agreed_roles(ctx)
        gt_roles = {role for role, row in ctx.ground_truth.items() if _usable(row)}
        if len(agreed) >= 2 or len(gt_roles) < 2:
            return None

        excluded = sorted((set(ctx.ground_truth) | set(ctx.recovery_result)) - agreed)
        return self._result(
            score=0.0,
            detail={
                "agreedRoles": sorted(agreed),
                "excludedRoles": excluded,
                "groundTruthRoles": len(gt_roles),
                "sharedInteractions": 0,
                "interactionOverlap": 0.0,
                "orderAgreement": 0.0,
            },
            per_role=[],
        )

    def _derivation_detail(self, inputs: Level3Inputs, ctx: ScoringContext) -> dict:
        """Which roles were walked, what each side found, and where each walk began."""
        return {
            #: The conditioning: roles both sides detected, and the ones that
            #: were consequently left out of both walks.
            "agreedRoles": sorted(inputs.agreed),
            "excludedRoles": inputs.excluded,
            "interactionsGroundTruth": len(inputs.gt_tokens),
            "interactionsRecoveryResult": len(inputs.rr_tokens),
            "sharedInteractions": len(inputs.shared),
            "unionSize": len(inputs.union),
            "entryRoleGroundTruth": inputs.ground_truth.entry_role,
            "entryRoleRecoveryResult": inputs.recovery_result.entry_role,
            #: Which rule chose each entry point — "behaviour" when the pattern's
            #: opening speaker is also a root of the call graph, "code-root" when
            #: it is not and the graph decided, "" when no root existed and no
            #: order was derived. A reviewer reading a surprising order should
            #: look here first.
            "entryRuleGroundTruth": inputs.ground_truth.entry_rule,
            "entryRuleRecoveryResult": inputs.recovery_result.entry_rule,
        }

    # ── shared reporting ───────────────────────────────────────────────────────

    def _rows(
        self,
        gt_tokens: list[str],
        rr_tokens: list[str],
        ground_truth: DetectionResult,
        recovery_result: DetectionResult,
        tokens: set[str] | None = None,
    ) -> list[dict]:
        """One row per interaction, ordered by ground truth then recovery.

        ``tokens`` narrows which interactions get a row; by default every
        interaction either side found gets one. Ranks stay the side's *own* ranks
        either way: restricting the rows must not renumber the walk.
        """
        gt_rank = {t: i + 1 for i, t in enumerate(gt_tokens)}
        rr_rank = {t: i + 1 for i, t in enumerate(rr_tokens)}
        gt_evidence = {i.token: i for i in ground_truth.interactions}
        rr_evidence = {i.token: i for i in recovery_result.interactions}

        # "Reordered" must mean *relative* to the other shared interactions, not
        # "sits at a different index". Inserting one interaction at the front shifts
        # every absolute rank by one while disturbing no relative order at all — and
        # Kendall's Tau, which counts pairs, would rightly call that concordant. A row
        # label that disagreed with the score above it would be worse than no label.
        shared = sorted(set(gt_tokens) & set(rr_tokens))

        def discordant_with_peers(token: str) -> int:
            return sum(
                1
                for other in shared
                if other != token
                and (gt_rank[token] < gt_rank[other]) != (rr_rank[token] < rr_rank[other])
            )

        wanted = set(gt_tokens) | set(rr_tokens) if tokens is None else set(tokens)
        rows = []
        for token in sorted(
            wanted,
            key=lambda t: (gt_rank.get(t, 10**6), rr_rank.get(t, 10**6), t),
        ):
            g, r = gt_rank.get(token), rr_rank.get(token)
            discordant = None
            if g is None:
                outcome = "onlyRecoveryResult"
            elif r is None:
                outcome = "onlyGroundTruth"
            else:
                discordant = discordant_with_peers(token)
                outcome = "agreed" if discordant == 0 else "reordered"

            source = gt_evidence.get(token) or rr_evidence.get(token)
            rows.append(
                {
                    "interaction": token,
                    "from": source.source if source else "",
                    "to": source.target if source else "",
                    "groundTruthRank": g,
                    "recoveryResultRank": r,
                    "outcome": outcome,
                    #: Shared interactions this one is ordered differently against.
                    #: None for a one-sided interaction, which has no relative order.
                    "discordantWith": discordant,
                    "evidence": {
                        "groundTruth": _evidence(gt_evidence.get(token)),
                        "recoveryResult": _evidence(rr_evidence.get(token)),
                    },
                }
            )
        return rows

    def _prescribed_steps(self, ctx: ScoringContext) -> int:
        return sum(1 for s in ctx.behaviour if (s.get("kind") or "").strip().lower() == "action")

    def _covered_steps(self, ctx: ScoringContext, union: set[str]) -> int:
        """How many prescribed *call* steps the detector found on either side.

        Informational only — it does not enter the score. Returns steps whose
        ``from → to`` pair was detected; return-kind steps are excluded because the
        detector does not model returns.
        """
        covered = 0
        for step in ctx.behaviour:
            if (step.get("kind") or "").strip().lower() != "action":
                continue
            token = f"{(step.get('from') or '').strip()} → {(step.get('to') or '').strip()}"
            if token in union:
                covered += 1
        return covered

# ── shared helpers ─────────────────────────────────────────────────────────────


def _conditioned(mappings: dict[str, dict], agreed: set[str]) -> dict[str, dict]:
    """One side's mappings, narrowed to the roles both sides detected.

    Dropping a role here is not the same as blanking it. It ceases to be an
    anchor, so the walk treats its entity as an ordinary function and passes
    straight through — the interactions on the far side of it are still found,
    now attributed to the nearest agreed role. That is what keeps a Level 1
    disagreement from tearing a hole in the order.
    """
    return {role: mapping for role, mapping in mappings.items() if role in agreed}


def _detect(ctx: ScoringContext, mappings: dict[str, dict]) -> DetectionResult:
    """One side's ordering, from the caller's detector.

    Both sides go through the same one, which is the point: an order compared
    against one derived by a different analysis measures the two analyses, not
    the recovery.
    """
    assert ctx.detect_order is not None  # guarded in _detect_inputs
    return ctx.detect_order(mappings, ctx.behaviour)


def _evidence(interaction) -> dict | None:
    if interaction is None:
        return None
    return {"file": interaction.file, "line": interaction.line, "via": list(interaction.via)}
