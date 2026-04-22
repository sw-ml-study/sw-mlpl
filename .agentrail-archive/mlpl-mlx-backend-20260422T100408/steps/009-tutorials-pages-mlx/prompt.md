Phase 5 step 009: tutorial lesson + using-mlx.md retrospective.

Make Saga 14 visible to users.

1. New web REPL tutorial lesson **"Running on MLX"** that walks:
   (a) `device("cpu") { randn(7, [1024, 64]) }` baseline
       forward pass.
   (b) Same expression wrapped in `device("mlx") { ... }`
       with the identical output (shape, labels, and, within
       tolerance, values).
   (c) A tiny training-loop swap that shows
       `device("mlx") { train N { adam(...) } }` producing
       the same loss curve shape as the CPU version but
       faster.
   If MLX is unavailable in the web REPL (wasm32), keep the
   lesson as text + screenshots showing the CLI run.
2. Update `docs/using-mlx.md` from "design sketch" to
   "reference": strip the `> Status: planned` disclaimer,
   align the API examples with what actually shipped, and add
   a retrospective section summarizing what landed vs what
   the sketch predicted.
3. Update `docs/status.md` one-liner.
4. Update `docs/saga.md` Saga 14 entry (move from "NEXT" to
   the completed-sagas list with a paragraph retrospective in
   the Saga 13 pattern).
5. Update `docs/plan.md` Saga 14 entry to `COMPLETE, v0.11.0`
   and its dependency graph.
6. Rebuild `pages/` via `scripts/build-pages.sh`; commit
   source and built artifact in the same commit.
7. `markdown-checker -f "**/*.md"` must pass (ASCII-only).
8. Quality gates + `/mw-cp`. Commit message references step 009.
