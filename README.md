# secpat-scoring

The three evaluation metrics of the paper *A Hybrid Approach for Automated
Security Pattern Recovery* (Section IV-C, Evaluation Metrics): role detection
accuracy (RDA), mapping localization score (MLS) and interaction recovery
correctness (IRC). Each compares the recovery of one recovery instance with the
groundtruth of the same codebase. The code calls them Level 1, 2 and 3, and also
reports `drift = 1 - score`.

The paper's Appendix A-D computes all three on a small example. The appendix is at
<https://doi.org/10.5281/zenodo.23664769>, and its Appendix G lists the other
artifacts. [secpatrec](https://github.com/anonymousspottedtoad/secpatrec) runs
these metrics on the paper's dataset (`secpat evaluate`).

One package, so every consumer computes the *same* numbers — two copies of the
metrics would make any divergence between them read as recovery drift when it was
really implementation drift.

## What is in here

| Paper | Code | What it scores | Module |
| ----- | ---- | -------------- | ------ |
| RDA | Level 1 — Role Detection | the share of the pattern's roles whose status (`present`, `external`, `missing`) agrees with the groundtruth | `level1_detection.py` |
| MLS | Level 2 — Role Grounding | for each role both sides found, `1 - d / 2H`, where `d` is the number of edges between the two mapped symbols through their lowest common ancestor in the codebase's containment tree and `H` is the tree's height; the mean over those roles | `level2_grounding.py`, `scoring_functions.py`, `containment/` |
| IRC | Level 3 — Interaction Order | the Jaccard overlap of the two interaction sets, times the share of pairs of shared interactions that both sides put in the same order (the paper's τ) | `level3_order.py` |

The orders Level 3 compares are derived by **`secpat-interaction-order`**, which this
package installs. `secpat_scoring.workspace.build_index` builds the struct index Level 2
reads.

`evaluate(EvaluationRequest) -> EvaluationReport` in `evaluation.py` is the single entry
point.

### What the paper's formulas leave out

The paper gives the formulas. The code also settles these cases:

- **Which roles count.** Only the roles the pattern requires: a role that the
  pattern's `04_pattern.json` marks `required: false` (Subject and System, or
  Entity and System in rate limiting) is left out of all three levels.
- **RDA.** A role with no row counts as `missing`, a row with no status as
  `present`, and the status `insecure` as `present`.
- **MLS.** A groundtruth role mapped to no symbol is skipped, and a recovered role
  mapped to no symbol scores 0. When a symbol has no place in the containment tree
  (no usable file, as for a library call), the role scores 1 if the two names are
  equal and 0 otherwise. MLS is 0 when no role is found by both sides.
- **IRC.** Only the roles both sides detected take part, and both sequences are
  derived over that role set with the same walk. The groundtruth mapping is walked
  again rather than its stored sequence read, so the groundtruth sequence compared
  can be shorter than the stored one. A role left out is walked through like any
  helper function. IRC is 0 when fewer than two roles were detected by both sides,
  and unavailable when the groundtruth itself has fewer than two. For the one-shot
  baseline, secpatrec scores the sequence the model stated instead of a walked one
  (`secpat evaluate --recovery-order stored`).

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
an index, so both URLs go in. Both repositories are public; no credentials are needed.

## Parser pinning

`tree-sitter` is pinned exactly, and grammars come one package per language (the bundled
`tree_sitter_languages` is unmaintained and stuck on tree-sitter 0.21.x). A grammar built
against another ABI fails by parsing every file to zero symbols rather than by raising,
which would read downstream as drift, so a consumer must not pin tree-sitter itself.

Two ordering details in `containment/ast_index.py` are load-bearing: `Language.query()`
was removed in 0.25 (use `Query` + `QueryCursor`), and captures come back grouped by
pattern rather than in document order — `_extract_calls` sorts them back by `start_byte`,
because only `calls[:10]` is kept and the order therefore decides what gets recorded.

## Checking an install

The repository has no test suite. To check an install against known numbers, score
one record of the paper's dataset with secpatrec, as its README's "Check the setup"
does: Level 1 `0.857`, Level 2 `0.986` and Level 3 `1.000` for `scality_osis`.

## License

MIT — see [LICENSE](LICENSE).
