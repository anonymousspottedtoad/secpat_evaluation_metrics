"""Pure scoring functions for Level 2 (Role Grounding).

Kept next to the metric implementations rather than inside ``containment/`` because
these are the *formulas*, not the tree construction — a new Level 2 formula would
add functions here and a new ``PatternMetric`` class beside it.

Tree-Induced Error through the lowest common ancestor, normalised by the hierarchy
diameter ``2H`` (Dekel 2004; Lanus 2025) — continuous in ``[0, 1]`` with exact match
at the top of the scale::

    d(a, b) = depth(a) + depth(b) - 2 * depth(LCA(a, b))
    score   = 1 - d / (2H)
"""

from __future__ import annotations

from .containment.containment_tree import Node, lca


def tree_induced_error(a: Node, b: Node) -> int | None:
    """Number of edges between two placed nodes through their LCA, or None."""
    anc = lca(a, b)
    if anc is None:
        return None
    return a.depth + b.depth - 2 * anc.depth


def path_length_score(a: Node, b: Node, height: int) -> float:
    """Normalised Path Length similarity in ``[0, 1]`` (1.0 == exact match).

    Falls back to string identity when a node is unplaced (no usable file): 1.0 if
    the two external names are equal, else 0.0.
    """
    if not a.placed or not b.placed:
        name_a = a.external_name or a.name
        name_b = b.external_name or b.name
        return 1.0 if name_a == name_b and name_a else 0.0
    d = tree_induced_error(a, b)
    if d is None:
        return 0.0
    diameter = 2 * max(height, 1)
    return max(0.0, 1.0 - d / diameter)
