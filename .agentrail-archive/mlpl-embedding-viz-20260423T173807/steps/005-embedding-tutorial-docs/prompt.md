Phase 3 step 005: web tutorial lesson +
`docs/using-embeddings.md`.

1. Add an "Embedding exploration" lesson to
   `apps/mlpl-web/src/lessons.rs`. Mirror the Saga 15
   "LoRA Fine-Tuning" pattern:
   - Short narrative (~4-5 sentences): what embeddings
     are, what `pairwise_sqdist` / `knn` / `tsne` do,
     what the 3-D scatter shows.
   - Interactive examples using a TINY fixture (4-8
     hand-constructed 2-D or 3-D points) so the WASM
     REPL stays responsive. t-SNE at N=8 is nearly
     instant; scatter3d at N=8 renders a readable
     plot.
   - `try_it` that suggests re-running with a
     different perplexity or a different seed and
     watching the layout change.
   - Keep the lesson under the
     `apps/mlpl-web/src/lessons.rs` 500-LOC sw-checklist
     budget. Use the same tactics Saga 15 used
     (semicolon-combined statements, trim redundant
     probes) if the initial draft tips over.

2. Rebuild `pages/` via
   `./scripts/build-pages.sh` and commit both source
   and the new WASM bundle.

3. Write `docs/using-embeddings.md` covering:
   - **What embeddings are + why we visualize them**
     (1 paragraph).
   - **The three new builtins**
     (`pairwise_sqdist`, `knn`, `tsne`) with one-line
     purpose each and a pointer to the contract.
   - **PCA is a composition pattern**, not a builtin --
     link to Saga 8's tutorial lesson and explain the
     power-iteration-on-covariance idiom. Explicitly
     state why we did not ship `pca` as a builtin
     (expressible in MLPL today; no per-primitive
     win).
   - **3-D scatter surface**: orthographic projection,
     fixed azimuth/elevation, no rotation. Pointer to
     the contract for the projection formula.
   - **Demo walkthrough** for `demos/embedding_viz.mlpl`
     (CPU). CLI command + what to watch for.
   - **Parity testing**: list the three new test
     files (`pairwise_sqdist_knn_tests.rs`,
     `tsne_tests.rs`, `embedding_viz_tests.rs`) and
     what each pins.
   - **Not shipped** (deferred follow-ups): UMAP, `pca`
     builtin, RAG pipeline (pending Saga 19), interactive
     3-D with rotation, MLX dispatch for t-SNE's inner
     loop.
   - **Related**: contracts, demo, tutorial lesson,
     sibling `docs/using-perturbation.md` and
     `docs/using-lora.md`.

4. `markdown-checker -f "docs/using-embeddings.md"`.

5. Quality gates + `/mw-cp`. Commit message references
   Saga 16 step 005.
