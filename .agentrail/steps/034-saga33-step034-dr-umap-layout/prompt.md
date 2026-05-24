Phase 2b of the dimensionality reduction milestone (full plan: docs/milestone-dimensionality-reduction.md). UMAP's optimization layer + the public wrapper. Depends on Phase 2a.

DELIVERABLES:

1. Layout optimization: 2D coordinate SGD on UMAP's cross-entropy + repulsion objective. Reuses the existing t-SNE optimization loop where the math is parallel (gradient descent over pairwise affinities); cleanly diverges at the loss function.

2. `umap(X, n_neighbors, min_dist, iters, seed)` builtin that wraps Phase 2a (knn_graph + fuzzy simplicial) and Phase 2b (layout SGD) into one call. Returns `[N, 2]`. Matches `tsne`'s API shape.

UNIT TESTS: end-to-end on a small fixture; deterministic given seed; output is bounded; loss decreases over iterations.

GRADCHECK: gradient via autodiff vs the analytical loss; tolerance matched to the t-SNE gradcheck baseline.

GLOSSARY: "UMAP" entry covers Riemannian-geometry + fuzzy-simplicial-sets framing per the milestone doc.

LANG-REFERENCE: new builtin row.

OUT OF SCOPE: comparison demos (Phase 3) and the tutorial lesson (Phase 4); standalone REPL invocation is enough verification for this step.
