Phase 3 step 006: release v0.14.0.

Cut the Embedding Visualization release.

1. Bump `Cargo.toml` `workspace.package.version` from
   `0.13.0` to `0.14.0`. All workspace members inherit
   via `version.workspace = true`; no per-crate edits
   required.

2. Update `CHANGELOG.md` with a v0.14.0 section above
   the v0.13.0 entry:
   - New builtins: `pairwise_sqdist`, `knn`, `tsne`.
   - New viz type: `svg(pts, "scatter3d")`.
   - New demo: `demos/embedding_viz.mlpl`.
   - New docs: `docs/using-embeddings.md` + tutorial
     lesson.
   - Scope notes: PCA is composition-only (not a
     builtin); MLX dispatch for `tsne` deferred;
     UMAP deferred; RAG pipeline deferred (pending
     Saga 19).

3. Insert Saga 16 retrospective entry in
   `docs/saga.md` above Saga 15 (newest-first
   convention). Summarize the shipped surface, the
   measured behavior of t-SNE on the demo fixture,
   and the deferred follow-ups.

4. Update `docs/status.md`:
   - Move Saga 16 row from Planned to Completed.
   - Roll remaining Planned target versions forward
     by one minor as necessary (v0.14 -> v0.15 etc).
   - Add explicit Planned rows for UMAP + `pca`
     builtin if the shipped doc lists them as
     deferred follow-ups, so the deferred work stays
     visible (same pattern used for QLoRA in the Saga
     15 release).

5. `cargo build --release` to confirm the bump
   compiles cleanly. Do NOT run `sw-install`
   (project memory `feedback_no_sw_install.md`).

6. `/mw-cp` quality gates: cargo test, clippy, fmt,
   markdown-checker on the updated docs, sw-checklist.

7. Tag `v0.14.0` locally. Do NOT push the tag unless
   the user confirms (matching the v0.12.0 / v0.13.0
   cadence).

8. `agentrail complete --done` closes the saga.
