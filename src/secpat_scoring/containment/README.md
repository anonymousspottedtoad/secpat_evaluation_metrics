# `containment/` — the AST index and containment tree

The one parser behind `struct.json` and the containment tree Level 2 is measured
over. A recovery tool that indexes a codebase can import `ast_index.parse_source`
rather than keep its own copy, so the `struct.json` a recovery runs against and the
index it is scored over come from one parser. Nothing outside this directory should
reimplement containment logic. `tests/test_scoring_level2.py` pins the Python
output (`fixtures/level2_expected.json`) and `tests/test_ast_index_java.py` pins the
Java shape.

**Python and Java** are parsed with tree-sitter. Java types nest (an inner class or
enum is a child of its class and carries its own methods, so `_add_symbol` in
`containment_tree.py` recurses), spans are the real declaration extent, an annotated
method opens at its first annotation — the line Joern's JAVASRC frontend anchors it
on — and class/method annotations are recorded as `decorators`. Interfaces, records
and annotation types are summarised as `kind: "class"`, so the kinds stay the four
Python produces. JS/TS/Go fall back to a regex scan of top-level symbols; `.kt` maps
to `java` but stays on the regex scan.

**The caller supplies the source text** — `parse_source(source, rel_path)` — so
where a codebase lives (disk, object storage) is the caller's business.

**Grammar loading.** `tree-sitter` is pinned exactly (0.26.0) with one PyPI grammar
package per language (`Language(m.language())`, `Parser(lang)`). Two ordering
details are load-bearing: `Language.query()` was removed in 0.25 (use `Query(lang,
src)` + `QueryCursor`), and captures come back grouped by pattern rather than in
document order — `_extract_calls` sorts them back by `start_byte`, because
`_symbol_summary` keeps only `calls[:10]` and the order therefore decides which
calls get recorded.

`containment_tree.py` reads only `name`, `kind`, `start_line`, `end_line` and
`children` from the struct index; `calls` does not affect Level 2.
