# `pca` Contract (Saga 16.5 step 001)

## Purpose

`pca(X, k) -> Y` reduces a rank-2 `[N, D]` input to
rank-2 `[N, k]` along its top-`k` principal axes.
Delivered as a one-line convenience builtin so callers
don't have to hand-roll the power-iteration + deflation
composition every time they want a PCA projection.

## Signature

```
pca(X, k) -> Y
```

- `X` -- rank-2 float array `[N, D]`. All entries must be
  finite.
- `k` -- positive integer, `1 <= k <= D`.
- `Y` -- rank-2 float array `[N, k]` of centered-and-
  projected data.

The `k` components themselves (eigenvectors) are NOT
returned as a separate output. Callers who need them can
run the power-iteration composition pattern directly --
see the Saga 8 "Dimensionality Reduction: PCA" tutorial
lesson for the under-the-hood view. Keeping the return
type a single rank-2 array keeps the builtin surface
small.

## Algorithm

1. Column-center: `Xc[i, j] = X[i, j] - mean_i(X[i, j])`.
2. Covariance: `Cov = Xc^T @ Xc / N`, `[D, D]` symmetric.
3. For component index `c = 0, 1, ..., k-1`:
   - Seed `v` at a standard basis vector not colinear
     with any already-extracted component; Gram-Schmidt
     orthogonalize against priors and normalize.
   - Power-iterate `v = normalize(Cov @ v)` for
     `POWER_ITERS = 50` steps, Gram-Schmidt against
     prior components at each step (guarantees
     orthogonality even when later eigenvalues are
     numerical noise).
   - Record `v` as component `c`.
   - `lambda = v^T Cov v` (Rayleigh quotient).
   - Deflate: `Cov -= lambda * (v outer v)`.
4. Stack components into `V [k, D]` row-major; return
   `Y = Xc @ V^T [N, k]`.

### Why Gram-Schmidt + deflation, not just deflation

For well-separated eigenvalues, power iteration on the
deflated covariance converges to the next dominant
eigenvector on its own. But when later eigenvalues are
small (numerical noise after `k-1` deflations on a rank-
deficient input, or near-zero variance along some axis),
deflation alone lets `v` drift back toward already-
extracted components. Gram-Schmidt inside the power-
iteration loop pins every new component into the
orthogonal complement of the prior ones, so `V` stays
orthonormal to machine precision regardless of spectral
gap.

## Hyperparameters

- `POWER_ITERS = 50`. Comfortably convergent at the
  matrix sizes we expect (`D` up to a few hundred) while
  keeping the algorithm cheap and deterministic.

Not user-tunable. A follow-up can expose them via a
config struct if concrete demand surfaces.

## Determinism

Two calls with identical `(X, k)` produce bit-identical
`Y`. The algorithm is fully deterministic: the seed
basis vector for each component is the standard basis
vector at index `(comp + start) % D` for the smallest
`start` that survives Gram-Schmidt with nonzero norm; no
PRNG is used.

## Sign ambiguity

Each extracted component `v` is unique only up to sign
(`v` and `-v` are both valid unit eigenvectors for the
same eigenvalue). With a deterministic basis seed the
sign is fixed for any given input, but is sensitive to
row-ordering changes in `X` that would affect the
covariance structure. Tests therefore pin variance-
preservation and shape claims, not absolute sign.

## Error cases

All errors surface as
`RuntimeError::InvalidArgument { func: "pca", reason }`
(or `ArityMismatch` for wrong arity), propagated as
`EvalError::RuntimeError(...)`.

- **Wrong arity.** Anything other than 2 arguments.
- **Non-rank-2 `X`.** "X must be rank-2 [N, D], got rank
  N".
- **Non-scalar `k`.** "k must be a scalar".
- **Non-positive or non-integer `k`.** "k must be a
  positive integer, got <value>".
- **`k > D`.** "k = <k> must be <= D = <D>".
- **Non-finite `X` entries.** "X must contain only
  finite values (no NaN/Inf)".

## What this contract does NOT cover

- **Returning the eigenvectors.** The builtin returns
  only the projected data `Y`. Callers who need the
  components can run the power-iteration composition
  pattern directly (Saga 8 lesson).
- **Returning eigenvalues / explained-variance ratios.**
  Not exposed; compute from column variances of `Y` if
  needed (`variance(Y[:, c]) = lambda_c`).
- **SVD-based PCA.** Not implemented; power iteration +
  deflation is the documented algorithm.
- **Sparse PCA, kernel PCA, randomized PCA.** Separate
  algorithms with different contracts; not shipped.
- **MLX dispatch.** Current implementation is CPU-only.
  The `D x D` covariance loop could dispatch to MLX for
  large `D`, but at embedding-table scale (D < 1024)
  launch overhead dominates. Deferred until a concrete
  large-`D` use case surfaces.
- **Incremental / streaming PCA.** The builtin takes the
  full matrix at once. Streaming APIs are a separate
  surface.
