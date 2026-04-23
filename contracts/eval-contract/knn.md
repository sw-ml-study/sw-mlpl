# `knn` Contract (Saga 16 step 001)

## Purpose

`knn(X, k) -> idx` returns the indices of the `k`
nearest non-self neighbors of every row in a rank-2
input `X [N, D]`, measured by squared Euclidean
distance. Used by the Saga 16 embedding-visualization
demos to answer "what's near this token?" questions.

## Signature

```
knn(X, k) -> idx
```

- `X` -- rank-2 float array `[N, D]`.
- `k` -- positive integer scalar, `1 <= k < N`.
- `idx` -- rank-2 float array `[N, k]` of integer-
  valued indices in `[0, N)`.

## Ordering

- **Primary order**: ascending squared Euclidean
  distance.
- **Tie-break**: lower original index first. The
  implementation is a stable sort on
  `(distance, index)` pairs, so identical distances
  preserve the original index order.
- **Self-exclusion**: row `i`'s own index `i` never
  appears in `idx[i]`. This is why `k < N` rather
  than `k <= N`.

## Error cases

All errors are `RuntimeError::InvalidArgument { func:
"knn", reason }`, propagated to the evaluator as
`EvalError::RuntimeError(...)`, except arity which
is `RuntimeError::ArityMismatch`.

- **Wrong arity.** Anything other than 2 arguments.
- **Non-rank-2 input.** "X must be rank-2 [N, D], got
  rank N".
- **Non-scalar k.** "k must be a scalar".
- **Non-integer or non-positive k.** "k must be a
  positive integer, got <value>".
- **k >= N.** "k = K must be < N = N (self is
  excluded, leaving N-1 candidates)".

## NaN and non-finite inputs

`partial_cmp` on `f64` returns `None` for NaN; the
sort treats that as equal and the stable tie-break
preserves the original index order. Callers who care
about NaN should filter `X` before calling `knn`.

## What this contract does NOT cover

- **Distances, not just indices.** `knn` returns
  indices only. If the distances are needed, compute
  `pairwise_sqdist(X)` and index into the result
  using the returned idx matrix.
- **Approximate / streaming k-NN.** Saga 16 ships an
  exact `O(N^2 * D + N^2 * log N)` algorithm. For
  large embedding tables a tree-based or
  approximate-graph index is a follow-up.
- **Cross-point-set queries.** `knn(X, Y, k)` with a
  separate query set `Y` is a natural extension;
  deferred until there is a concrete demand.
- **Self-inclusive variant.** Use case driven
  (PyTorch's `topk` over pairwise distances does
  include self); add a flag if requested.
