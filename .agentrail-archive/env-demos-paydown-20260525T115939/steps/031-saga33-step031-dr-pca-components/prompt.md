Phase 1a of the dimensionality reduction milestone (full plan: docs/milestone-dimensionality-reduction.md).

DELIVERABLE: New `pca_components(X, k)` builtin in `crates/mlpl-runtime-dim-reduction/src/pca_builtin.rs`. Returns the `[k, D]` loadings matrix where row i is the i-th principal component direction in original feature space. The existing `pca(X, k)` computes these internally via power iteration + deflation and discards them; this step surfaces them.

Sister return: per-component variance-explained percentages as a `[k]` vector via a parallel `pca_variance_explained(X, k)` builtin (or fold both into one return, designer's call).

UNIT TESTS in tests/: shape assertions ([k, D]), orthonormality of returned rows, variance-explained sums to <= total variance.

GLOSSARY: brief entry on "Principal components / Loadings" if not present.

LANG-REFERENCE: new builtin row(s).

OUT OF SCOPE: the viz that consumes the loadings (that is Phase 1b, the next step).
