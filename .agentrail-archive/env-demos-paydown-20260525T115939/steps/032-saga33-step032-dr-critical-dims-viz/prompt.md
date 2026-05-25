Phase 1b of the dimensionality reduction milestone (full plan: docs/milestone-dimensionality-reduction.md). Depends on Phase 1a (`pca_components` builtin).

DELIVERABLE: New viz type `svg(loadings, "critical_dimensions", feature_names)` in `crates/mlpl-viz/`. Takes a `[k, D]` loadings matrix and an optional `[D]` string list of feature names. Renders a horizontal heatmap with D columns and k rows; cell (i, j) colored by |loadings[i, j]| (magnitude of feature j's contribution to component i). Top-N most "critical" features per component highlighted (bordered or asterisked). Right-side legend has per-component variance-explained percentages.

NEW DEMO (or extend the existing PCA demo): runs `pca_components(X, 3)` on a synthetic dataset with named features, then renders the critical-dimensions heatmap. The reading is "Component 1 is mostly feature 7. Component 2 is mostly feature 12 and 13."

OUT OF SCOPE: the permutation-sensitivity variant for t-SNE / UMAP (deferred to Phase 4 lesson because UMAP doesn't ship until Phase 2). The viz primitive supports both inputs equally; only the data-generation pipeline differs.

QUALITY GATES include pages rebuild + a mlpl-reg regression test for the new viz.
