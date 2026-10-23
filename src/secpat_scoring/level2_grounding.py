"""Level 2 — Role Grounding (component / entity correctness).

    "*Given the role was correctly detected*, did the recovery result cite the
    correct code artifact?"

Conditioned on Level 1: only roles that are a Level 1 true positive — detected in
*both* the ground truth and the recovery result — are scored here, so a missed or
hallucinated role is never penalised twice.

Every role is grounded to a named entity, scored by normalised Path Length over
the Tree-Induced Error through the containment hierarchy's lowest common
ancestor — continuous in ``[0, 1]``, exact match being the top of the scale
rather than a separate metric. A method cited as its enclosing class, or as a
sibling method, takes a small penalty instead of the perfect score that entity
canonicalisation used to award.

A detected role that cites no entity is not a grounding at all. On the ground
truth side that is an annotation defect, so the role is skipped rather than
charged to the recovery; on the recovery side it scores 0.

**No true positive scores 0.** When the recovery detected none of the roles the
ground truth grounds, it grounded nothing correctly, and Level 2 is 0 rather than
unavailable — the miss is charged here as well as in Level 1, deliberately (the
same rule as Level 3 below two agreed roles). It stays unavailable only when the
ground truth itself grounds no role, so there is nothing a recovery could match.

A different Level 2 formula (Wu-Palmer, or a TSED-blended variant) would be a new
class here with the same ``key``.
"""

from __future__ import annotations

from .base import MetricResult, PatternMetric, ScoringContext, is_detected
from .containment.containment_tree import ContainmentTree
from .scoring_functions import path_length_score, tree_induced_error


class PathLengthGrounding(PatternMetric):
    """Mean per-role grounding score, by path length through the containment tree."""

    key = "level2"
    level = "Level 2 — Role Grounding"
    formula = "path-length"
    headline_label = "Mean grounding score"

    def score(self, ctx: ScoringContext) -> MetricResult:
        if ctx.tree is None:
            return self._unavailable()

        height = ctx.tree.height()
        per_role: list[dict] = []
        skipped = 0
        total = 0.0

        for role in sorted(ctx.ground_truth):
            g = ctx.ground_truth[role]
            p = ctx.recovery_result.get(role)

            # Level 1 gate — score only roles detected on both sides.
            if not is_detected(g) or not is_detected(p) or not (g.get("entity") or "").strip():
                skipped += 1
                continue

            row = self._score_role(role, g, p, ctx.tree, height)
            total += row["score"]
            per_role.append(row)

        if not per_role:
            groundable = [
                role for role, g in ctx.ground_truth.items()
                if is_detected(g) and (g.get("entity") or "").strip()
            ]
            if not groundable:
                return self._unavailable()
            return self._result(
                score=0.0,
                detail={
                    "aggregate": 0.0,
                    "scoredRoles": 0,
                    "skippedRoles": skipped,
                    "groundTruthRoles": len(groundable),
                    "treeHeight": height,
                },
                per_role=[],
            )

        aggregate = total / len(per_role)
        return self._result(
            score=aggregate,
            detail={
                "aggregate": round(aggregate, 4),
                "scoredRoles": len(per_role),
                "skippedRoles": skipped,
                "treeHeight": height,
            },
            per_role=per_role,
        )

    # -- per-role scoring -------------------------------------------------------

    def _score_role(
        self, role: str, g: dict, p: dict, tree: ContainmentTree, height: int
    ) -> dict:
        g_entity = (g.get("entity") or "").strip()
        p_entity = (p.get("entity") or "").strip()

        if not p_entity:
            return {
                "patternRole": role,
                "score": 0.0,
                "distance": None,
                "groundTruth": {"entity": g_entity, "loc": _loc(g)},
                "recoveryResult": {"entity": "", "loc": _loc(p)},
            }

        g_node = tree.resolve(g_entity, g.get("file", ""), normalise_lines(g.get("lines")))
        p_node = tree.resolve(p_entity, p.get("file", ""), normalise_lines(p.get("lines")))
        base = path_length_score(g_node, p_node, height)
        distance = (
            tree_induced_error(g_node, p_node) if g_node.placed and p_node.placed else None
        )

        return {
            "patternRole": role,
            "score": round(base, 4),
            "distance": distance,
            "groundTruth": {"entity": g_entity, "loc": _loc(g)},
            "recoveryResult": {"entity": p_entity, "loc": _loc(p)},
        }


#: A definition span may be written with a typographic dash ("77–88") while the
#: line parser only splits on ASCII "-". Normalise before parsing so the anchor
#: line is read from the start of the span.
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-"})


def normalise_lines(lines: str | None) -> str:
    return (lines or "").strip().translate(_DASHES)


def _loc(m: dict) -> str:
    f, ln = m.get("file", ""), m.get("lines", "")
    return f"{f}:{ln}" if f and ln else (f or "")
