# secpat-scoring

Pattern drift scoring: how far a recovered security-pattern mapping drifts from a
ground-truth mapping of the same codebase, at three levels. One package, so every
consumer computes the *same* numbers — two copies of the metrics would make any
divergence between them read as recovery drift when it was really implementation
drift.

## What is in here

| Level | Metric | Module |
| ----- | ------ | ------ |
| 1 | Role detection | `level1_detection.py` |
| 2 | Role grounding (containment distance) | `level2_grounding.py`, `containment/` |
| 3 | Interaction order: overlap (Jaccard) × Kendall tau over shared interactions | `level3_order.py` |

The orders Level 3 compares are derived by **`secpat-interaction-order`**, which this
package installs. `secpat_scoring.workspace.build_index` builds the struct index Level 2
reads.

`evaluate(EvaluationRequest) -> EvaluationReport` in `evaluation.py` is the single entry
point. `drift = 1 - score`.

## What is deliberately *not* in here

Acquiring the inputs. The package is handed rows, a struct index and an order detector;
it never fetches a repo or a ground truth. Where they come from is the caller's business.

Deriving an interaction order is not in here either. The caller binds
`secpat_interaction_order.detect` to a repo and passes it in as `detect_order`; Level 3
calls it once per side and compares the two results.

## Install

```bash
pip install \
  "secpat-interaction-order @ git+https://github.com/anonymousspottedtoad/secpat_interaction_recovery.git@v0.1.0" \
  "secpat-scoring @ git+https://github.com/anonymousspottedtoad/secpat_evaluation_metrics.git@v0.23.0"
```

`secpat-scoring` pins `secpat-interaction-order` by name, and pip cannot resolve that from
an index, so both URLs go in.

The repo is **private**, so this needs GitHub credentials. For local work,
`gh auth login && gh auth setup-git` makes the plain https URL resolve; in Docker, pass a
token as a BuildKit secret rather than baking it into a layer.

## Parser pinning

`tree-sitter` is pinned exactly, and grammars come one package per language (the bundled
`tree_sitter_languages` is unmaintained and stuck on tree-sitter 0.21.x). A grammar built
against another ABI fails by parsing every file to zero symbols rather than by raising,
which would read downstream as drift, so a consumer must not pin tree-sitter itself.

Two ordering details in `containment/ast_index.py` are load-bearing: `Language.query()`
was removed in 0.25 (use `Query` + `QueryCursor`), and captures come back grouped by
pattern rather than in document order — `_extract_calls` sorts them back by `start_byte`,
because only `calls[:10]` is kept and the order therefore decides what gets recorded.

## Tests

```bash
pip install -e ../secpat-interaction-order -e ".[dev]"
pytest
```

`tests/fixtures/level2_expected.json` pins Level 2 to known-good numbers on a small
sample app; it is the regression check for the containment index and the formula.
