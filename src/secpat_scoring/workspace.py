"""The struct index a repo needs before Levels 2 and 3 can read it.

All this module does now is parse a checkout into the ``struct`` index — file
path -> symbol summary, with line ranges. Level 2 consumes it as a containment
tree; Level 3 reads nothing from it at all any more.

It used to own ``make_backend`` as well: the fall-through chain of backends, the
Joern export and its cache, and the ``IndirectBackend`` wrapper that supplied the
invocations a call graph cannot contain. All of that is gone with the Python
walk. The interaction order is derived on the query service now — see
``secpat_interaction_order`` — so the only thing a caller still builds locally
is this index. It was ``secpat_scoring.sequence.workspace`` until 0.23.0, when
the detector moved out into that package and this stayed: it is built from
``containment``, and the detector never reads it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from .containment.ast_index import language_for, parse_source, tree_to_summary

logger = logging.getLogger(__name__)

#: Directory names that hold tooling state or vendored third-party code rather
#: than the codebase under analysis.
BLOCKED_DIRS: frozenset[str] = frozenset(
    ".git node_modules __pycache__ .venv venv dist build target vendor"
    " .eggs .mypy_cache .ruff_cache .pytest_cache".split()
)


def is_blocked(path: Path, root: Path) -> bool:
    """True if any parent directory of ``path`` should be excluded from analysis.

    Hidden directories are excluded wholesale: they hold tooling state, not the
    codebase — a tool's own index folder (``.kb/``) would otherwise be parsed as
    though it were source.

    This is the default filter for :func:`build_index`; a caller that walks the
    repo itself can import it so every walk sees one file set. A caller with its
    own policy may pass one.
    """
    for part in path.relative_to(root).parts[:-1]:
        if part in BLOCKED_DIRS or part.endswith(".egg-info"):
            return True
        if part.startswith(".") and part not in (".", ".."):
            return True
    return False


def build_index(
    codebase: Path,
    language: str = "python",
    *,
    is_blocked: Callable[[Path, Path], bool] | None = is_blocked,
) -> dict[str, dict]:
    """Parse ``codebase`` into the ``struct`` index Levels 2 and 3 read.

    Level 2 reads it as a containment tree; Level 3 reads line ranges from it.
    Keyed by repo-relative path, so a role citation lines up against the same key
    either way.

    It used to return a second index of call sites with line numbers, for the
    tree-sitter backend to resolve calls against. Nothing reads it now — the
    export the walk runs on carries its own call sites, resolved rather than
    matched by name.

    Only files :func:`~..containment.ast_index.language_for` calls ``language``
    are read.
    """
    struct: dict[str, dict] = {}
    for path in sorted(codebase.rglob("*")):
        if not path.is_file() or (is_blocked is not None and is_blocked(path, codebase)):
            continue
        rel = str(path.relative_to(codebase))
        if language_for(rel) != language:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tree = parse_source(source, rel)
        if tree is not None:
            struct[rel] = tree_to_summary(tree)
    return struct


__all__ = ["BLOCKED_DIRS", "build_index", "is_blocked"]
