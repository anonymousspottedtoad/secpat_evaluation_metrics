"""Containment hierarchy for Level 2 (Role Grounding) scoring.

It consumes only the struct-index dict — see ``README.md`` in this directory.

Builds a single tree per codebase from the persisted struct index (the AST
symbol index) plus the file paths::

    codebase-root (depth 0)
    └── directory segment(s)   arionxiv/ , arionxiv/services/
        └── file               unified_auth_service.py
            └── top-level sym   class UnifiedAuthenticationService | function verify_token
                └── child sym   method authenticate_user

Directory segments are included so that two entities in the same package share a
deeper lowest-common-ancestor than two in unrelated packages, making the
Tree-Induced Error meaningfully graded across files (not just within one class).
A chain of single-child directories is merged into one node, so a level that
separates nothing (Java's ``src/main/java/com/acme``) does not inflate the height
or the distances.

A citation ``(entity, file, lines)`` is mapped onto a node by :meth:`ContainmentTree.resolve`.
Library / unresolved names that are not in the AST are attached as *synthetic*
leaves under the cited file so cross-file distances stay well-defined; a citation
with no file at all is returned *unplaced* (scored by string identity upstream).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass
class Node:
    """A node in the containment tree.

    ``depth`` is the number of edges from the root (root itself is depth 0). A node
    is *placed* when it belongs to the built tree (root/dir/file/symbol or a
    synthetic leaf under a file); an *unplaced* node carries only ``external_name``
    and is compared by string identity instead of tree distance.
    """

    name: str
    # "root" | "dir" | "file" | "class" | "enum" | "function" | "method" | "external"
    kind: str = ""
    depth: int = 0
    parent: Node | None = None
    children: dict[str, Node] = field(default_factory=dict)
    start_line: int = 0
    end_line: int = 0
    placed: bool = True
    external_name: str = ""

    def child(self, key: str) -> Node | None:
        return self.children.get(key)

    def add_child(self, key: str, node: Node) -> Node:
        node.parent = self
        node.depth = self.depth + 1
        self.children[key] = node
        return node


def _norm_path(file: str) -> str:
    """Normalise a citation's file path to the ``struct.json`` key form."""
    p = (file or "").strip().replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _add_symbol(parent: Node, sym: dict, default_kind: str = "function") -> None:
    node = Node(
        name=sym.get("name", ""),
        kind=sym.get("kind", default_kind),
        start_line=int(sym.get("start_line", 0) or 0),
        end_line=int(sym.get("end_line", 0) or 0),
    )
    # Symbols keyed by name; on a rare name clash within one file the later one wins
    # for dict lookup, but resolution below scans *all* descendants by line anyway.
    parent.add_child(f"sym:{node.name}:{node.start_line}", node)
    # Recursive: a Java nested type is a child of its class and has methods of its
    # own. Python's index never goes deeper than class -> method.
    for child_sym in sym.get("children", []) or []:
        _add_symbol(node, child_sym, "method")


class ContainmentTree:
    def __init__(self, struct: dict[str, dict]):
        self.root = Node(name="<root>", kind="root", depth=0)
        self._file_nodes: dict[str, Node] = {}
        for path, tree in struct.items():
            file_node = self._ensure_path(_norm_path(path))
            for sym in tree.get("symbols", []) or []:
                _add_symbol(file_node, sym)
        self._collapse_chains(self.root)
        self._set_depths(self.root, 0)
        # Height of the built AST tree (before any synthetic leaves are attached).
        self._height = self._compute_height(self.root)

    # ── construction ──────────────────────────────────────────────────────────

    def _ensure_path(self, path: str) -> Node:
        parts = PurePosixPath(path).parts
        cur = self.root
        for i, part in enumerate(parts):
            kind = "file" if i == len(parts) - 1 else "dir"
            key = f"{kind}:{part}"
            nxt = cur.child(key)
            if nxt is None:
                nxt = cur.add_child(key, Node(name=part, kind=kind))
            cur = nxt
        self._file_nodes[path] = cur
        return cur

    @staticmethod
    def _collapse_chains(node: Node) -> None:
        """Merge every directory whose only child is a directory into that child.

        Such a level separates nothing — every node below it shares it — yet it
        adds one to the tree height that normalises Level 2, and one to the
        distance of any pair it lies between. Java's package path
        (``src/main/java/com/acme/…``) is mostly such levels, so without this the
        same citation error scores higher in a Java repo than in a Python one.
        The root absorbs a sole top-level directory the same way.
        """
        while (
            node.kind in ("root", "dir")
            and len(node.children) == 1
            and next(iter(node.children.values())).kind == "dir"
        ):
            (only,) = node.children.values()
            if node.kind == "dir":
                node.name = f"{node.name}/{only.name}"
            node.children = only.children
            for child in node.children.values():
                child.parent = node
        for child in node.children.values():
            ContainmentTree._collapse_chains(child)

    @staticmethod
    def _set_depths(node: Node, depth: int) -> None:
        node.depth = depth
        for child in node.children.values():
            ContainmentTree._set_depths(child, depth + 1)

    @staticmethod
    def _compute_height(node: Node) -> int:
        if not node.children:
            return node.depth
        return max(ContainmentTree._compute_height(c) for c in node.children.values())

    def height(self) -> int:
        return max(self._height, 1)

    # ── lookup ────────────────────────────────────────────────────────────────

    def _file_node(self, file: str) -> Node | None:
        path = _norm_path(file)
        if not path:
            return None
        node = self._file_nodes.get(path)
        if node is not None:
            return node
        # Fall back to a basename match (citation path may be rooted differently).
        base = PurePosixPath(path).name
        matches = [n for p, n in self._file_nodes.items() if PurePosixPath(p).name == base]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _descendant_symbols(file_node: Node):
        stack = list(file_node.children.values())
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def resolve(self, entity: str, file: str, lines: str) -> Node | None:
        """Map a ``(entity, file, lines)`` citation to a tree node.

        Returns ``None`` when the citation names no entity — there is nothing to
        place, and Level 2 never scores such a row. Otherwise always returns a Node: a real
        AST node when found, a synthetic leaf under the file when the name is a
        library/unresolved symbol, or an *unplaced* external node when there is no
        usable file.
        """
        entity = (entity or "").strip()
        if not entity:
            return None

        line = _parse_line(lines)
        file_node = self._file_node(file)

        if file_node is not None:
            match = self._match_symbol(file_node, entity, line)
            if match is not None:
                return match
            # Library/unresolved name with a known file: synthetic leaf at the line.
            leaf = Node(name=entity, kind="external", start_line=line or 0, end_line=line or 0)
            file_node.add_child(f"ext:{entity}:{line}", leaf)
            return leaf

        # No usable file: unplaced, compared by string identity upstream.
        return Node(name=entity, kind="external", placed=False, external_name=entity)

    @staticmethod
    def _match_symbol(file_node: Node, entity: str, line: int | None) -> Node | None:
        candidates: list[Node] = []
        for n in ContainmentTree._descendant_symbols(file_node):
            if n.name == entity or n.name.split(".")[-1] == entity.split(".")[-1]:
                candidates.append(n)
        if not candidates:
            return None
        if line is not None:
            candidates.sort(key=lambda n: abs(n.start_line - line))
        return candidates[0]


def _parse_line(lines: str) -> int | None:
    """Extract the anchor line number from a ``lines`` field (``"12"`` or ``"12-40"``)."""
    s = (lines or "").strip()
    if not s:
        return None
    head = s.split("-")[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def lca(a: Node, b: Node) -> Node | None:
    """Lowest common ancestor of two placed nodes in the same tree, or None."""
    if a is None or b is None or not a.placed or not b.placed:
        return None
    ancestors: set[int] = set()
    cur: Node | None = a
    while cur is not None:
        ancestors.add(id(cur))
        cur = cur.parent
    cur = b
    while cur is not None:
        if id(cur) in ancestors:
            return cur
        cur = cur.parent
    return None
