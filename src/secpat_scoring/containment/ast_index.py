"""AST symbol-tree builder — the struct index Level 2 scoring is measured over.

The one parser behind ``struct.json``: a recovery tool can import it rather than
keep a copy, so the index a recovery is run against and the one it is scored over
are the same document. See ``README.md`` in this directory.

Python and Java are parsed properly with tree-sitter (classes, methods, params,
calls, decorators/annotations, docstrings/Javadoc, base classes; Java types nest,
so an inner class carries its own methods). Other languages fall back to a regex
scan that recovers top-level classes and functions only, each a fixed 30-line
span — enough to place a file in the hierarchy, not to bound a symbol.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class SymbolNode:
    name: str
    qualified_name: str
    kind: str                            # "class" | "enum" | "function" | "method"
    start_line: int
    end_line: int
    code: str = ""
    docstring: str = ""
    base_classes: list[str] = field(default_factory=list)   # classes / enums only
    params: list[str] = field(default_factory=list)          # functions / methods only
    calls: list[str] = field(default_factory=list)           # functions / methods only
    decorators: list[str] = field(default_factory=list)      # methods; Java types too (annotations)
    children: list[SymbolNode] = field(default_factory=list)  # methods (Java: and nested types) inside a class


@dataclass
class FileTree:
    path: str
    language: str
    symbols: list[SymbolNode] = field(default_factory=list)


# ── Struct-index helpers ──────────────────────────────────────────────────────


def _symbol_summary(sym: SymbolNode) -> dict:
    s: dict = {
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "kind": sym.kind,
        "start_line": sym.start_line,
        "end_line": sym.end_line,
    }
    if sym.docstring:
        s["docstring"] = sym.docstring
    if sym.base_classes:
        s["base_classes"] = sym.base_classes
    if sym.params:
        s["params"] = sym.params
    if sym.calls:
        s["calls"] = sym.calls[:10]
    if sym.children:
        s["children"] = [_symbol_summary(c) for c in sym.children]
    return s


def tree_to_summary(tree: FileTree) -> dict:
    """Return a JSON-serialisable summary of a FileTree (no source code)."""
    return {
        "path": tree.path,
        "language": tree.language,
        "symbols": [_symbol_summary(s) for s in tree.symbols],
    }


# ── Language registry ─────────────────────────────────────────────────────────
#
# Grammars come from one PyPI package per language. The bundled
# `tree_sitter_languages` package this
# used to import is stuck on tree-sitter 0.21.x and unmaintained; keeping it
# would have forced consumers onto incompatible parsers, and a shared scoring
# package can only pin one.

_LANG_SUFFIXES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "java",
    ".go": "go",
}

#: Language indexed when a caller names none. A repo is recovered in one
#: language at a time.
DEFAULT_LANGUAGE: str = "python"


#: Grammar module and factory for each language name. `typescript` is the odd
#: one out: its package ships two grammars and names them per-dialect.
_GRAMMAR_MODULES: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
}


@cache
def _get_language(lang_name: str):
    """Return the tree-sitter Language for a language name, or None.

    Cached: building a Language is pure and the struct index calls this once per
    file, so a repo of any size would otherwise rebuild the same few grammars
    thousands of times.
    """
    entry = _GRAMMAR_MODULES.get(lang_name)
    if entry is None:
        return None
    module_name, factory = entry
    try:
        from importlib import import_module

        from tree_sitter import Language

        module = import_module(module_name)
        return Language(getattr(module, factory)())
    except Exception as e:  # pragma: no cover - grammar load failure
        logger.warning("AST: language load for %s failed: %s", lang_name, e)
        return None


def _get_parser(lang_name: str):
    """Return a tree-sitter Parser bound to ``lang_name``, or None."""
    lang = _get_language(lang_name)
    if lang is None:
        return None
    try:
        from tree_sitter import Parser

        return Parser(lang)
    except Exception as e:  # pragma: no cover - parser construction failure
        logger.warning("AST: parser load for %s failed: %s", lang_name, e)
        return None


# ── Python parser ─────────────────────────────────────────────────────────────

_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _first_docstring(body_node, source: bytes) -> str:
    for child in body_node.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type in ("string", "concatenated_string"):
                    raw = _node_text(sub, source).strip("'\"\r\n ")
                    for delim in ('"""', "'''", '"', "'"):
                        if raw.startswith(delim):
                            raw = raw[len(delim) :]
                            break
                    for delim in ('"""', "'''", '"', "'"):
                        if raw.endswith(delim):
                            raw = raw[: -len(delim)]
                            break
                    return raw.splitlines()[0].strip() if raw else ""
        break
    return ""


#: Call names inside a function body: bare `foo()` and attribute `x.foo()`.
_CALL_QUERY = "(call function: [(identifier) @c (attribute attribute: (identifier) @c)])"


@cache
def _compiled_call_query(lang, query: str):
    """Compile a call query once per grammar — parsing it is not free."""
    from tree_sitter import Query

    return Query(lang, query)


def _extract_calls(func_node, lang, source: bytes, query: str = _CALL_QUERY) -> list[str]:
    try:
        from tree_sitter import QueryCursor

        captures = QueryCursor(_compiled_call_query(lang, query)).captures(func_node)
        # The query has two alternatives, and >=0.22 returns matches grouped by
        # pattern rather than in document order. Sort back into source order:
        # `_symbol_summary` keeps only `calls[:10]`, so the order decides which
        # calls are recorded at all, and struct.json has to stay byte-compatible
        # across runs.
        nodes = sorted(captures.get("c", []), key=lambda n: (n.start_byte, n.end_byte))
        seen: dict[str, None] = {}
        for n in nodes:
            t = _node_text(n, source)
            if t not in seen:
                seen[t] = None
        return list(seen)
    except Exception:
        return []


def _extract_params(params_node, source: bytes) -> list[str]:
    skip = {"self", "cls"}
    result = []
    for child in params_node.children:
        if child.type == "identifier":
            name = _node_text(child, source)
            if name not in skip:
                result.append(name)
        elif child.type in ("typed_parameter", "default_parameter", "typed_default_parameter"):
            for sub in child.children:
                if sub.type == "identifier":
                    name = _node_text(sub, source)
                    if name not in skip:
                        result.append(name)
                    break
    return result


def _parse_func_node(node, source: bytes, lang, class_name: str | None) -> SymbolNode:
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    body_node = node.child_by_field_name("body")

    func_name = _node_text(name_node, source) if name_node else "unknown"
    qualified = f"{class_name}.{func_name}" if class_name else func_name

    decorators: list[str] = []
    for sibling in (node.parent.children if node.parent else []):
        if sibling.id == node.id:
            break
        if sibling.type == "decorator":
            decorators.append(_node_text(sibling, source).strip())

    return SymbolNode(
        name=func_name,
        qualified_name=qualified,
        kind="method" if class_name else "function",
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        code=_node_text(node, source),
        docstring=_first_docstring(body_node, source) if body_node else "",
        params=_extract_params(params_node, source) if params_node else [],
        calls=_extract_calls(node, lang, source),
        decorators=decorators,
    )


def _parse_class_node(node, source: bytes, lang) -> SymbolNode:
    name_node = node.child_by_field_name("name")
    class_name = _node_text(name_node, source) if name_node else "Unknown"
    body_node = node.child_by_field_name("body")

    args_node = node.child_by_field_name("superclasses")
    base_classes: list[str] = []
    if args_node:
        for child in args_node.children:
            if child.type == "identifier":
                base_classes.append(_node_text(child, source))

    is_enum = any(b in _ENUM_BASES for b in base_classes)

    methods: list[SymbolNode] = []
    if body_node and not is_enum:
        for child in body_node.children:
            inner = child
            if child.type == "decorated_definition":
                inner = child.child_by_field_name("definition") or child
            if inner.type == "function_definition":
                methods.append(_parse_func_node(inner, source, lang, class_name))

    return SymbolNode(
        name=class_name,
        qualified_name=class_name,
        kind="enum" if is_enum else "class",
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        code=_node_text(node, source),
        docstring=_first_docstring(body_node, source) if body_node else "",
        base_classes=base_classes,
        children=methods,
    )


def _parse_python(rel_path: str, source: bytes, lang, parser) -> FileTree:
    root = parser.parse(source).root_node
    symbols: list[SymbolNode] = []

    for node in root.children:
        inner = node
        if node.type == "decorated_definition":
            inner = node.child_by_field_name("definition") or node

        if inner.type == "function_definition":
            symbols.append(_parse_func_node(inner, source, lang, None))
        elif inner.type == "class_definition":
            symbols.append(_parse_class_node(inner, source, lang))

    return FileTree(path=rel_path, language="python", symbols=symbols)


# ── Java parser ───────────────────────────────────────────────────────────────

#: Type declarations, and the kind each is summarised as. The struct index keeps
#: the kinds Python produces, so an interface, a record and an annotation type all
#: read as a ``class`` to the containment tree and the chunker.
_JAVA_TYPE_KINDS: dict[str, str] = {
    "class_declaration": "class",
    "interface_declaration": "class",
    "record_declaration": "class",
    "annotation_type_declaration": "class",
    "enum_declaration": "enum",
}

#: Members summarised as methods. A record's compact constructor and an annotation
#: type's element (``String value();``) are both methods for this purpose.
_JAVA_METHOD_TYPES = frozenset({
    "method_declaration",
    "constructor_declaration",
    "compact_constructor_declaration",
    "annotation_type_element_declaration",
})

#: ``extends`` on a class, ``implements`` on a class/enum/record, ``extends`` on an interface.
_JAVA_SUPERTYPE_CLAUSES = frozenset({"superclass", "super_interfaces", "extends_interfaces"})

#: Call names inside a method body: `foo()` / `x.foo()`, and `new Foo()` — the
#: counterpart of Python's `Foo()`, which the Python query captures as a call.
_JAVA_CALL_QUERY = (
    "(method_invocation name: (identifier) @c)"
    " (object_creation_expression type: [(type_identifier) @c (generic_type (type_identifier) @c)])"
)


def _java_doc(node, source: bytes) -> str:
    """First sentence line of the Javadoc directly above a declaration, or ``""``."""
    prev = node.prev_named_sibling
    if prev is None or prev.type != "block_comment":
        return ""
    raw = _node_text(prev, source)
    if not raw.startswith("/**"):
        return ""
    for line in raw[3:].removesuffix("*/").splitlines():
        line = line.strip().lstrip("*").strip()
        if line and not line.startswith("@"):
            return line
    return ""


def _java_annotations(node, source: bytes) -> list[str]:
    for child in node.children:
        if child.type == "modifiers":
            return [
                _node_text(a, source)
                for a in child.children
                if a.type in ("marker_annotation", "annotation")
            ]
    return []


def _java_supertypes(node, source: bytes) -> list[str]:
    """Supertype names with type arguments dropped: ``Base<T>`` -> ``Base``."""
    names: list[str] = []
    for clause in node.children:
        if clause.type not in _JAVA_SUPERTYPE_CLAUSES:
            continue
        for t in clause.named_children:
            for ty in t.named_children if t.type == "type_list" else [t]:
                if ty.type == "generic_type" and ty.named_children:
                    ty = ty.named_children[0]
                names.append(_node_text(ty, source))
    return names


def _java_params(params_node, source: bytes) -> list[str]:
    result = []
    for p in params_node.named_children:
        if p.type == "formal_parameter":
            name = p.child_by_field_name("name")
        elif p.type == "spread_parameter":
            decl = next((c for c in p.named_children if c.type == "variable_declarator"), None)
            name = decl.child_by_field_name("name") if decl else None
        else:  # receiver parameter (`Foo this`), comments
            continue
        if name is not None:
            result.append(_node_text(name, source))
    return result


def _java_members(body):
    """A type body's members; an enum's sit behind its constants, in one more node."""
    for child in body.named_children:
        if child.type == "enum_body_declarations":
            yield from child.named_children
        else:
            yield child


def _parse_java_method(node, source: bytes, lang, owner: str) -> SymbolNode:
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    name = _node_text(name_node, source) if name_node else "unknown"
    return SymbolNode(
        name=name,
        qualified_name=f"{owner}.{name}",
        kind="method",
        # A method's node opens at its first annotation, which is also the line
        # JAVASRC anchors it on (see `secpat_interaction_order/joern/java.sc`); Javadoc is a
        # separate comment node and is not included.
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        code=_node_text(node, source),
        docstring=_java_doc(node, source),
        params=_java_params(params_node, source) if params_node else [],
        calls=_extract_calls(node, lang, source, _JAVA_CALL_QUERY),
        decorators=_java_annotations(node, source),
    )


def _parse_java_type(node, source: bytes, lang, outer: str | None) -> SymbolNode:
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node, source) if name_node else "Unknown"
    qualified = f"{outer}.{name}" if outer else name
    body = node.child_by_field_name("body")

    children: list[SymbolNode] = []
    for member in _java_members(body) if body else ():
        if member.type in _JAVA_TYPE_KINDS:
            children.append(_parse_java_type(member, source, lang, qualified))
        elif member.type in _JAVA_METHOD_TYPES:
            children.append(_parse_java_method(member, source, lang, qualified))

    return SymbolNode(
        name=name,
        qualified_name=qualified,
        kind=_JAVA_TYPE_KINDS[node.type],
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        code=_node_text(node, source),
        docstring=_java_doc(node, source),
        base_classes=_java_supertypes(node, source),
        decorators=_java_annotations(node, source),
        children=children,
    )


def _parse_java(rel_path: str, source: bytes, lang, parser) -> FileTree:
    root = parser.parse(source).root_node
    symbols = [
        _parse_java_type(node, source, lang, None)
        for node in root.children
        if node.type in _JAVA_TYPE_KINDS
    ]
    return FileTree(path=rel_path, language="java", symbols=symbols)


# ── Generic parser (regex-based) ──────────────────────────────────────────────

_GENERIC_PATTERNS: dict[str, dict[str, re.Pattern]] = {
    "javascript": {
        "class": re.compile(r"^\s*(?:export\s+)?class\s+(\w+)", re.MULTILINE),
        "function": re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE
        ),
    },
    "typescript": {
        "class": re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE),
        "function": re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE
        ),
    },
    "java": {
        "class": re.compile(
            r"^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)",
            re.MULTILINE,
        ),
        "function": re.compile(
            r"^\s+(?:(?:public|private|protected|static|final|synchronized|abstract|override)\s+)*"
            r"\w[\w<>\[\]]*\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
    },
    "go": {
        "class": re.compile(r"^type\s+(\w+)\s+struct\b", re.MULTILINE),
        "function": re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", re.MULTILINE),
    },
}


def _parse_generic(rel_path: str, source: bytes, lang_name: str) -> FileTree:
    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines()
    patterns = _GENERIC_PATTERNS.get(lang_name, {})
    symbols: list[SymbolNode] = []

    for kind_label, pat in patterns.items():
        for m in pat.finditer(text):
            name = m.group(1)
            start_line = text[: m.start()].count("\n") + 1
            end_line = min(start_line + 30, len(lines))
            kind = "class" if kind_label == "class" else "function"
            code = "\n".join(lines[start_line - 1 : end_line])
            symbols.append(
                SymbolNode(
                    name=name,
                    qualified_name=name,
                    kind=kind,
                    start_line=start_line,
                    end_line=end_line,
                    code=code,
                )
            )

    return FileTree(path=rel_path, language=lang_name, symbols=symbols)


# ── Public API ────────────────────────────────────────────────────────────────


#: Suffixes parsed with tree-sitter, and the grammar and parser for each; every
#: other supported suffix falls back to the regex scan. Keyed by suffix rather
#: than language because `.kt` is indexed as "java" but the Java grammar cannot
#: parse Kotlin.
_TREE_SITTER_PARSERS = {
    ".py": ("python", _parse_python),
    ".java": ("java", _parse_java),
}


def language_for(rel_path: str) -> str | None:
    """Language name for a path's extension, or None when unsupported."""
    return _LANG_SUFFIXES.get(PurePosixPath(rel_path).suffix.lower())


def parse_source(source: str, rel_path: str) -> FileTree | None:
    """Parse source text into a SymbolNode tree. None when the language is unsupported.

    The caller passes the text in, so where the codebase lives (disk, object
    storage) is the caller's business.
    """
    lang_name = language_for(rel_path)
    if lang_name is None:
        return None

    raw = source.encode("utf-8", errors="replace")

    grammar = _TREE_SITTER_PARSERS.get(PurePosixPath(rel_path).suffix.lower())
    if grammar is not None:
        grammar_name, parse = grammar
        lang = _get_language(grammar_name)
        parser = _get_parser(grammar_name)
        if lang is not None and parser is not None:
            try:
                return parse(rel_path, raw, lang, parser)
            except Exception as e:
                logger.warning("AST: tree-sitter failed on %s: %s", rel_path, e)

    return _parse_generic(rel_path, raw, lang_name)
