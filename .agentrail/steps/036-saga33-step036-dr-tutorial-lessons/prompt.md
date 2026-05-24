Phase 4 of the dimensionality reduction milestone (full plan: docs/milestone-dimensionality-reduction.md). Tutorial lessons in dependency order. Depends on Phases 1-3.

DELIVERABLES: six new tutorial lessons under `crates/mlpl-web-lessons/src/` (or wherever the existing tutorial lessons live). Each lesson is one markdown blob with embedded MLPL code snippets:

1. "Why reduce dimensions?" -- motivation (manifold hypothesis, curse of dimensionality, visualizing high-D). Concept-first, no code.

2. "PCA: the linear baseline" -- builds on the PCA demo, adds the loadings view via pca_components + svg(_, "critical_dimensions"). Positioned as "fastest answer, try first."

3. "SNE: the very-slow ancestor" -- description-only, no MLPL primitive. Walks through Hinton + Roweis 2002, KL divergence, the symmetrization-and-crowding problem. "Why it was abandoned" framing.

4. "t-SNE: a peek at nonlinear methods" -- the existing tsne builtin. Explains van der Maaten + Hinton 2008's two fixes. The "cluster shape is meaningful, distances between clusters are not" caveat.

5. "UMAP: the modern default" -- the headline lesson. Riemannian geometry + fuzzy simplicial sets (informal). Runs demos/umap_vs_pca.mlpl and demos/umap_vs_tsne.mlpl. Explains why UMAP preserves more global structure than t-SNE.

6. "Reading a critical-dimensions heatmap" -- centered on the viz itself. Same data viewed through PCA loadings, permutation sensitivity for t-SNE, permutation sensitivity for UMAP; teaches the user to read each.

Permutation-sensitivity tooling for lessons 4 and 6 may need a small helper builtin -- file as a sub-step if non-trivial.

PAGES REBUILD required.
