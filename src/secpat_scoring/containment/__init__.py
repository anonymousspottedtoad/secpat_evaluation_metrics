"""Vendored AST + containment-hierarchy code — see README.md for provenance."""

from .ast_index import DEFAULT_LANGUAGE, FileTree, SymbolNode, language_for, parse_source, tree_to_summary
from .containment_tree import ContainmentTree, Node, lca

__all__ = [
    "ContainmentTree",
    "DEFAULT_LANGUAGE",
    "FileTree",
    "Node",
    "SymbolNode",
    "language_for",
    "lca",
    "parse_source",
    "tree_to_summary",
]
