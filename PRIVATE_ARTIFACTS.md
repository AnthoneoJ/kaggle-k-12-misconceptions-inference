# Required private resources

These files are derived from the competition training data, so they are not committed (see `.gitignore`). Place them at these paths before running notebooks 02 and 05:

| File | Used by |
|---|---|
| `resources/corpus/library.jsonl` | stage 2 (annotated examples and review precedents) |
| `resources/corpus/cards.json` | stage 2 (rule cards) |
| `resources/corpus/partners.json` | stage 2 (usual confusions between codes) |
| `resources/corpus/library_stats.json` | stage 2 |
| `resources/reference/question_solution_cache.json` | pass 2 (reference worked solutions) |

Each file's expected sha256 is in `resources/expected_hashes.json`. Notebook 00 and each stage stop with a clear message if a file differs.
