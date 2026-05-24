Phase 5 of the dimensionality reduction milestone (full plan: docs/milestone-dimensionality-reduction.md). The "complete the survey" methods. Both ship behind unit tests; demos only inside the dim_reduction_zoo composite from Phase 3.

DELIVERABLES:

1. `mds(X, k, iters)` builtin. Multidimensional Scaling -- precompute pairwise distance matrix, find low-D coordinates that preserve those distances. Ship the SGD variant (simpler, reuses t-SNE / UMAP's optimization loop). Implementation in `crates/mlpl-runtime-dim-reduction/`.

2. `random_projection(X, k, seed)` builtin. Johnson-Lindenstrauss random projection: a random `[D, k]` matrix approximately preserves pairwise distances for modest k. One-liner implementation. The payoff is the "you don't always need principled methods" point in the "Why reduce dimensions?" lesson.

UNIT TESTS: shape assertions, deterministic given seed (for random_projection), distance-preservation parity check vs sklearn fixtures (captured as static fixtures -- NOT a sklearn build dep).

GLOSSARY: brief entries for MDS and Johnson-Lindenstrauss if not present.

LANG-REFERENCE + inspect_groups: new builtin rows.

OUT OF SCOPE: Kernel PCA, Isomap, NMF, autoencoder-based reduction (deferred per the milestone non-goals).

DEMO INTEGRATION: update demos/dim_reduction_zoo.mlpl (filed in Phase 3) to include the new methods if it didn't already.
