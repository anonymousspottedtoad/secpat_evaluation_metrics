"""Level 1 — Role Detection (present / external / missing).

    "Is Role R implemented in this codebase, fulfilled outside it, or absent —
    regardless of what evidence is cited?"

One label per (codebase, role) pair over a **fixed, known label space** — the
reference pattern's role vocabulary — from three classes (:data:`~.base.LEVEL1_CLASSES`).

Two formulas live here, both behind :class:`PatternMetric` with the same ``key``:

* :class:`RoleAgreementDetection` — **the active one.** Plain agreement rate: what
  fraction of roles the ground truth and the recovery result say the same thing
  about. A role both sides call absent counts as agreement, because it is.
* :class:`MacroF1RoleDetection` — the metrics summary's macro-F1 formulation, kept
  as the registered alternative. Not currently selected; see the class docstring
  for why it reads badly on a single codebase.

Swap between them in ``DEFAULT_METRICS`` (``secpat_scoring/__init__.py``).
"""

from __future__ import annotations

from .base import MetricResult, PatternMetric, ScoringContext, is_detected, level1_class

# The two outcomes a role can have at Level 1. Deliberately binary: the ground
# truth and recovery result either say the same thing about this role or they do
# not. *Which way* they differ stays visible in the two status columns beside it.
OUTCOME_AGREED = "agreed"
OUTCOME_MISMATCHED = "mismatched"


def _status_of(mapping: dict | None) -> str:
    if not mapping:
        return "missing"
    return (mapping.get("status") or "present").strip().lower() or "missing"


class RoleAgreementDetection(PatternMetric):
    """Fraction of roles the ground truth and the recovery result agree on.

        score = agreed roles / total roles in the vocabulary

    Agreement is on the role's Level 1 class (:func:`~.base.level1_class`):
    `present` (implemented in the repository — `insecure` included, since how well
    is Level 2/2.5's question), `external` (fulfilled by a library or service the
    repository only calls), or `missing`. So `present` vs `insecure` is agreement,
    `present` vs `external` is not, and **`missing` vs
    `missing` is agreement too** (both say it is not there). *(Before
    secpat-scoring 0.21.0, 2026-10-19, `external` agreed with `present`: Level 1
    asked only whether the role was detected.)*

    This is the plain reading of Level 1 and the one that survives being shown to a
    reviewer: the number on screen is exactly the proportion of the table's rows
    marked "agreed".
    """

    key = "level1"
    level = "Level 1 — Role Detection"
    formula = "agreement-rate"
    headline_label = "Roles agreed"

    def score(self, ctx: ScoringContext) -> MetricResult:
        vocabulary = list(ctx.role_vocabulary)
        if not vocabulary:
            return self._unavailable()

        per_role: list[dict] = []
        agreed = 0

        for role in vocabulary:
            in_truth = is_detected(ctx.ground_truth.get(role))
            in_result = is_detected(ctx.recovery_result.get(role))
            is_agreed = level1_class(ctx.ground_truth.get(role)) == level1_class(
                ctx.recovery_result.get(role)
            )
            agreed += int(is_agreed)

            per_role.append(
                {
                    "patternRole": role,
                    "groundTruth": _status_of(ctx.ground_truth.get(role)),
                    "recoveryResult": _status_of(ctx.recovery_result.get(role)),
                    "groundTruthDetected": in_truth,
                    "recoveryResultDetected": in_result,
                    "outcome": OUTCOME_AGREED if is_agreed else OUTCOME_MISMATCHED,
                }
            )

        rate = agreed / len(vocabulary)

        return self._result(
            score=rate,
            # Only what the header line needs; the table below carries the detail.
            detail={"agreedRoles": agreed, "totalRoles": len(vocabulary)},
            per_role=per_role,
        )


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class MacroF1RoleDetection(PatternMetric):
    """Macro-averaged F1 over the role vocabulary — the metrics summary's formula.

    Registered alternative, **not currently active**. Kept because it is the
    formulation `evaluation_metrics_summary.md` specifies and cites (Tsoumakas &
    Katakis 2007), and because it is the right aggregate once results are pooled
    across many codebases.

    Why it is not the default: on a *single* codebase every per-role F1 collapses
    to 0 or 1, and under the `zero_division=0` convention a role both sides agree
    is **absent** scores F1=0 — it has no true positives to divide by. So a
    perfectly matching recovery result can still report drift, which reads as a bug
    to a reviewer. `RoleAgreementDetection` counts that case as the agreement it is.
    """

    key = "level1"
    level = "Level 1 — Role Detection"
    formula = "macro-f1"
    headline_label = "Macro-F1"

    def score(self, ctx: ScoringContext) -> MetricResult:
        vocabulary = list(ctx.role_vocabulary)
        if not vocabulary:
            return self._unavailable()

        per_role: list[dict] = []
        f1_sum = 0.0
        truth_positive: set[str] = set()
        result_positive: set[str] = set()

        for role in vocabulary:
            g = is_detected(ctx.ground_truth.get(role))
            p = is_detected(ctx.recovery_result.get(role))
            if g:
                truth_positive.add(role)
            if p:
                result_positive.add(role)

            tp = int(g and p)
            fp = int(p and not g)
            fn = int(g and not p)
            tn = int(not g and not p)

            # scikit-learn's zero_division=0 convention: a zero denominator scores 0.
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = _f1(precision, recall)
            f1_sum += f1

            per_role.append(
                {
                    "patternRole": role,
                    "groundTruth": _status_of(ctx.ground_truth.get(role)),
                    "recoveryResult": _status_of(ctx.recovery_result.get(role)),
                    "groundTruthDetected": g,
                    "recoveryResultDetected": p,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "outcome": OUTCOME_AGREED if g == p else OUTCOME_MISMATCHED,
                }
            )

        macro_f1 = f1_sum / len(vocabulary)
        union = truth_positive | result_positive
        jaccard = len(truth_positive & result_positive) / len(union) if union else 1.0
        exact_match = 1.0 if truth_positive == result_positive else 0.0

        return self._result(
            score=macro_f1,
            detail={
                "macroF1": round(macro_f1, 4),
                "jaccard": round(jaccard, 4),
                "exactMatch": exact_match,
                "roleCount": len(vocabulary),
                "groundTruthDetectedCount": len(truth_positive),
                "recoveryResultDetectedCount": len(result_positive),
                "truePositives": len(truth_positive & result_positive),
                # Raw FN/FP: under-detection vs over-detection, which F1 conflates.
                "falseNegatives": len(truth_positive - result_positive),
                "falsePositives": len(result_positive - truth_positive),
            },
            per_role=per_role,
        )
