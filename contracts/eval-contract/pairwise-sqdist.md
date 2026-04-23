# `pairwise_sqdist` Contract (Saga 16 step 001)

## Purpose

`pairwise_sqdist(X) -> D` returns the `[N, N]` matrix of
squared Euclidean distances between every pair of rows in
a rank-2 input `X [N, D]`. Promotes the
`matmul + reduce_add + broadcasting` identity used
inline by every prior distance-based demo (K-Means,
Saga-8 PCA, Saga-16 k-NN and t-SNE) into a single
named surface.

## Signature

```
pairwise_sqdist(X) -> D
```

- `X` -- rank-2 float array `[N, D]`.
- `D` -- rank-2 float array `[N, N]` where
  `D[i, j] = sum over k in 0..D of (X[i, k] - X[j, k])^2`.

## Guarantees

- **Symmetry.** `D[i, j] == D[j, i]` elementwise.
- **Zero diagonal.** `D[i, i] == 0.0` for every `i`.
- **Shape round-trip.** An empty input (`[0, D]`)
  returns an empty `[0, 0]` result rather than
  panicking.

## Error cases

All errors are `RuntimeError::InvalidArgument { func:
"pairwise_sqdist", reason }`, propagated to the
evaluator as `EvalError::RuntimeError(...)`.

- **Wrong arity.** Anything other than 1 argument
  returns `RuntimeError::ArityMismatch`.
- **Non-rank-2 input.** "X must be rank-2 [N, D], got
  rank N".

## Implementation

CPU-only direct `O(N^2 * D)` loop today. The
`matmul + reduce_add` identity would inherit MLX
dispatch for free via `dispatched_call`, but at the
embedding-table sizes we care about (V <= few
thousand, D <= 512) the direct loop is faster or
equivalent. MLX dispatch is a deferred optimisation
follow-up; the numerical surface is unchanged.

## What this contract does NOT cover

- **L1, cosine, or Mahalanobis distance.** Only
  squared Euclidean today. Other metrics are a
  follow-up once a concrete demand surfaces.
- **Cross-product distances.** `pairwise_sqdist(X,
  Y)` between two different point sets is a natural
  extension but not shipped in Saga 16.
- **Explicit sqrt.** Callers who need Euclidean
  (non-squared) can compute `sqrt(pairwise_sqdist(X))`
  using the existing `sqrt` builtin; shipping a
  separate `pairwise_dist` builds nothing new.
