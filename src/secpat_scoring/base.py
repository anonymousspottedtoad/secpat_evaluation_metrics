"""The common interface every pattern-drift metric implements.

The evaluation framework defines several levels, and each level has more than one defensible formula. So a
*level* is not a function here — it is an interface, and each formula is a class
behind it. Swapping "Level 2 scored by normalised path length" for "Level 2 scored
by Wu-Palmer similarity" means adding a class and changing one entry in
``DEFAULT_METRICS``; no existing metric is edited.

Scoring direction, fixed across every metric:

    score  ∈ [0, 1]   1.0 == the recovery result agrees perfectly with the ground truth
    drift  = 1 - score

The two sides, named the same way everywhere:

* **ground truth** — the reviewer's finalised, conflict-resolved role mapping.
* **recovery result** — what the recovery pipeline produced, immutable.

So drift reads as "how far the recovery result strayed from what the review
concluded is really there".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from secpat_interaction_order import DetectionResult

from .containment.containment_tree import ContainmentTree

# The four status terms. `missing` is also what a role with no row at all
# counts as.
STATUS_PRESENT = "present"
STATUS_INSECURE = "insecure"
STATUS_EXTERNAL = "external"
STATUS_MISSING = "missing"

#: Statuses that count as "this role was detected in the codebase" for Level 1.
#: Detection asks only whether the role is there at all — *how well* it is
#: implemented is what Levels 2 and 2.5 answer, so `insecure` is a detection too.
DETECTED_STATUSES: frozenset[str] = frozenset({STATUS_PRESENT, STATUS_INSECURE, STATUS_EXTERNAL})


def is_detected(mapping: dict | None) -> bool:
    """True when a role row counts as a Level 1 positive.

    Over the four-term status vocabulary. A row with no status at all is treated as
    `present`: the pipeline emits rows it believes in, and older recovery results
    predate the status field.
    """
    if not mapping:
        return False
    status = (mapping.get("status") or STATUS_PRESENT).strip().lower()
    return status in DETECTED_STATUSES


#: The three classes Level 1 compares. `insecure` says the role is implemented in
#: the repository, so it is `present`; `external` (fulfilled by a
#: library or service the repository only calls) is a class of its own.
LEVEL1_CLASSES: tuple[str, ...] = (STATUS_PRESENT, STATUS_EXTERNAL, STATUS_MISSING)


def level1_class(mapping: dict | None) -> str:
    """The Level 1 class of a role row: `present`, `external` or `missing`.

    Same reading of a row as :func:`is_detected` — no row is `missing`, a row with
    no status is `present` — with the detected statuses split into in-repository
    (`present`, `insecure`) and `external`.
    """
    if not is_detected(mapping):
        return STATUS_MISSING
    status = (mapping.get("status") or STATUS_PRESENT).strip().lower()
    return STATUS_EXTERNAL if status == STATUS_EXTERNAL else STATUS_PRESENT


@dataclass(frozen=True)
class ScoringContext:
    """Everything a metric may need to score one (codebase, pattern) pair.

    A metric reads only the parts it needs and reports ``available=False`` when
    something it requires is absent — it must never raise.
    """

    #: role name -> role-mapping row. The ground truth.
    ground_truth: dict[str, dict]
    #: role name -> role-mapping row. The recovery's output — the side being
    #: scored. (In classification terms this is the *prediction*; it is called
    #: the recovery result everywhere, because that is what it is.)
    recovery_result: dict[str, dict]
    #: The fixed label space — the reference pattern's role names.
    role_vocabulary: list[str] = field(default_factory=list)
    #: Containment hierarchy for the codebase; None when the struct index is unbuilt.
    tree: ContainmentTree | None = None
    #: The reference pattern's numbered behaviour walkthrough (Level 3 input).
    behaviour: list[dict] = field(default_factory=list)
    #: The raw struct index (file path -> symbol summary). Level 3 needs the symbol
    #: line ranges directly; Level 2 consumes the same data through ``tree``.
    struct: dict[str, dict] | None = None
    #: Derives one side's interaction order from a role mapping and the
    #: walkthrough — normally ``secpat_interaction_order.detect`` bound to this repo and its
    #: language. Level 3 calls it once per side, with that side's mapping
    #: *conditioned* to the roles both sides detected, which is why the
    #: derivation cannot be hoisted out and done once by the caller. The
    #: walkthrough is ``behaviour`` above, handed over rather than bound by the
    #: caller, so the walk sees the steps the evaluation scores and no others.
    #: ``None`` means Level 3 reports itself unavailable.
    detect_order: Callable[[dict[str, dict], list[dict]], DetectionResult] | None = None
    #: Scratch space shared by the metrics of one scoring run, keyed by whoever
    #: owns the entry. Two metrics of the same level often need the same expensive
    #: derivation — every Level 3 formula walks the call graph from the same
    #: conditioned role set — and memoising it here means the walk runs once
    #: without either metric having to know the other exists. Excluded from
    #: comparison: it is a cache, not part of what the context *is*.
    cache: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass
class MetricResult:
    """One metric's verdict, in the shape the API and UI consume."""

    key: str
    level: str
    formula: str
    available: bool
    headline_label: str
    score: float | None = None
    #: Metric-level numbers (macro-F1, jaccard, aggregate, …).
    detail: dict = field(default_factory=dict)
    #: Per-role breakdown rows, rendered as a table in the reviewer UI.
    per_role: list[dict] = field(default_factory=list)

    @property
    def drift(self) -> float | None:
        return None if self.score is None else 1.0 - self.score

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "level": self.level,
            "formula": self.formula,
            "available": self.available,
            "headlineLabel": self.headline_label,
            "score": None if self.score is None else round(self.score, 4),
            "drift": None if self.drift is None else round(self.drift, 4),
            "detail": self.detail,
            "perRole": self.per_role,
        }


class PatternMetric(ABC):
    """One formula for one level of the evaluation framework."""

    #: Stable identifier used by the API and the UI — one per *level*, not per formula.
    key: ClassVar[str]
    #: Human-readable level name, e.g. "Level 1 — Role Detection".
    level: ClassVar[str]
    #: Which formula this class implements, e.g. "macro-f1". Distinguishes two
    #: implementations of the same `key`.
    formula: ClassVar[str]
    #: Label for the headline number, e.g. "Macro-F1".
    headline_label: ClassVar[str]

    @abstractmethod
    def score(self, ctx: ScoringContext) -> MetricResult:
        """Score one (codebase, pattern) pair. Must not raise — report unavailability."""

    # -- helpers for subclasses -------------------------------------------------

    def _unavailable(self) -> MetricResult:
        return MetricResult(
            key=self.key,
            level=self.level,
            formula=self.formula,
            available=False,
            headline_label=self.headline_label,
            score=None,
        )

    def _result(
        self,
        score: float,
        detail: dict,
        per_role: list[dict],
    ) -> MetricResult:
        return MetricResult(
            key=self.key,
            level=self.level,
            formula=self.formula,
            available=True,
            headline_label=self.headline_label,
            score=score,
            detail=detail,
            per_role=per_role,
        )
