Phase 2a of the dimensionality reduction milestone (full plan: docs/milestone-dimensionality-reduction.md). The UMAP implementation's data-structure layer.

DELIVERABLES:

1. `knn_graph(X, k)` builtin in `crates/mlpl-runtime-dim-reduction/`. Brute-force k-nearest-neighbor edge list: returns `[N*k, 3]` matrix with rows `(i, j, dist)`. N up to ~5000, k up to ~30 -- brute force fits the budget per the milestone non-goals.

2. Fuzzy simplicial set construction: local sigma estimation (binary search to match a target neighbor count) + symmetrization (`a + b - a*b` per the UMAP paper). Surface as a builtin `fuzzy_simplicial_set(neighbors, distances)` or fold into the higher-level wrapper (designer's call).

UNIT TESTS: brute-force k-NN parity against a hand-rolled distance matrix; sigma binary search converges; symmetrization is commutative.

OUT OF SCOPE: the layout optimization SGD and the public `umap()` wrapper. Those are Phase 2b.

NEW BUILTINS go in NAMES of the dim-reduction crate, in inspect_groups, in lang-reference.
