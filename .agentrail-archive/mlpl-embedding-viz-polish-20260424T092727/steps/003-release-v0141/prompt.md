Phase 2 step 003: demos + docs + release v0.14.1.

Integrate the two new builtins into the shipped
surface and cut a patch release.

1. **Update `demos/embedding_viz.mlpl`**:
   - Replace the column-selector matmul 3-D
     projection with `emb_3d = pca(table, 3)`. One
     line instead of the `reshape(...) + matmul`
     pair.
   - Update the inline comment to point at the new
     builtin and note that the column-selector
     shortcut is no longer needed.

2. **Update `docs/using-embeddings.md`**:
   - "PCA is a composition pattern, not a builtin"
     section: rewrite as "PCA is shipped as a builtin
     in v0.14.1". Keep the power-iteration recipe as
     pedagogical context (link to the Saga 8 lesson
     for the under-the-hood view). Update the
     one-line example in the text.
   - New "Extracting embed-layer weights" section:
     describe `embed_table(model)`, show the
     `train chain -> embed_table(chain) -> tsne ->
     svg` idiom, explain first-match semantics.
   - "Not shipped" list: remove the `pca(X, k)`
     builtin and the `embed_table(model)` builtin
     entries. Keep UMAP, RAG, interactive 3-D,
     MLX-for-tsne, Barnes-Hut.

3. **Update the "Embedding exploration" web REPL
   lesson** (`apps/mlpl-web/src/lessons_advanced.rs`):
   - Add 1 example showing `pca(X, 2)` alongside
     the existing `tsne` call so the user sees
     both reducers side by side.
   - Add 1 example showing `embed_table(...)` on a
     small chain.
   - Keep the lesson at a similar size; trim other
     examples if needed so the file stays under
     sw-checklist's 500-LOC budget.
   - Update try_it to mention the new builtins.

4. **Rebuild `pages/`** via
   `./scripts/build-pages.sh` and commit both
   source and the new WASM bundle.

5. **Release v0.14.1**:
   - `Cargo.toml` workspace.package.version
     `0.14.0 -> 0.14.1`. Patch-level bump (additive
     surface, no breaking change).
   - `CHANGELOG.md`: new v0.14.1 section above
     v0.14.0. Entry lists the two new builtins,
     the demo update, doc + lesson updates. Scope
     notes: UMAP still deferred; interactive 3-D
     still deferred; MLX-for-tsne still deferred.
   - `docs/saga.md`: fold Saga 16.5 as a short
     "Addendum" paragraph under the Saga 16
     retrospective entry -- it's an in-family
     follow-up, not a full saga.
   - `docs/status.md`: add a Saga 16.5 row in
     Completed (below Saga 16). Remove the
     `pca + embed_table` line from the deferred
     Planned rows.
   - `cargo build --release` to confirm the bump.
   - `/mw-cp` quality gates: cargo test, clippy,
     fmt, markdown-checker, sw-checklist.
   - Tag `v0.14.1` locally. Do NOT push unless the
     user confirms (v0.12.0 / v0.13.0 / v0.14.0
     cadence).

6. `agentrail complete --done` closes Saga 16.5.
