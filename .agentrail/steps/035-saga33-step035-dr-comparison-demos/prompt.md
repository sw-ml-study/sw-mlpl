Phase 3 of the dimensionality reduction milestone (full plan: docs/milestone-dimensionality-reduction.md). The three comparison demos. Depends on Phase 2 (UMAP) and indirectly Phase 1 (critical-dimensions viz for one demo).

DELIVERABLES: three new demos in `demos/` AND mirrored web playground entries in a new file `apps/mlpl-web/src/demos_dim_reduction.rs` (the home for all dim-reduction demos -- groups PCA, UMAP, and the comparisons):

1. `demos/umap_vs_pca.mlpl`: curved manifold (Swiss roll or two-moons embedded in higher D). PCA smears the manifold; UMAP recovers geometry. Renders both 2D embeddings side by side.

2. `demos/umap_vs_tsne.mlpl`: multi-cluster dataset where inter-cluster distance is meaningful. t-SNE preserves local structure; UMAP preserves both. Side-by-side scatters with a legend calling out which global feature each method preserved.

3. `demos/dim_reduction_zoo.mlpl`: PCA + t-SNE + UMAP + MDS + random projection on one dataset, rendered as a row of five thumbnails with a caption per method. Depends on Phase 5 MDS + random projection too -- file as a deliverable here but mark the MDS / random_projection inclusions with TODO until Phase 5 lands, OR sequence after Phase 5.

WEB DEMOS: add each demo (or curated subset) to demos_dim_reduction.rs; register in demos.rs's DEMOS slice.

GLOSSARY: brief entries for "Swiss roll" / "manifold preservation" if not present.

PAGES REBUILD required since apps/mlpl-web changed.
