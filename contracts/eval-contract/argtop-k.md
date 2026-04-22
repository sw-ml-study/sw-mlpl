# `argtop_k` Contract (Saga 20 step 003)

## Purpose

`argtop_k(values, k)` returns the `k` indices of the largest
entries in a rank-1 float vector. It is the index-returning
companion to the existing `top_k(logits, k)` (which masks
logits for sampling). The two builtins have different names
because their return types differ -- `top_k` returns masked
logits shaped like the input, `argtop_k` returns an index
vector shaped `[k]`.

Used by the Neural Thickets ensemble loop to pick the K
best-scoring variants out of a `losses : [N]` vector.

## Signature

```
argtop_k(values, k) -> indices : [k]
```

- `values` -- rank-1 float vector.
- `k` -- non-negative integer scalar, with `0 <= k <= len(values)`.
- Returns a rank-1 float vector of length `k` whose entries
  are integer-valued indices into `values`.

## Ordering

- Primary order: **descending** by value. The first returned
  index points at the largest entry.
- Tie-break: **lower original index first**. The sort is
  stable, so `argtop_k([1.0, 1.0, 0.0], 2) == [0, 1]`.
- Indices are NOT sorted by index. `argtop_k([0.9, 0.1, 0.5,
  0.2], 4) == [0, 2, 3, 1]`, which is the descending-value
  permutation of `0..4`.

## NaN and non-finite inputs

`partial_cmp` returns `None` for `NaN`; the sort treats
that as equal, which places `NaN` entries among the
"middle" of the ordering. Callers that care about NaN should
filter or replace before calling `argtop_k`. This matches
the existing `top_k` / `sample` conventions.

## Return type

`DenseArray` shaped `[k]` with integer-valued `f64` entries.
MLPL does not have a separate integer array type; the f64
convention is used consistently across `argmax`, `top_k`,
`sample`, and the Model DSL's token id outputs.

## Error cases

All errors are `RuntimeError::InvalidArgument { func:
"argtop_k", reason }`, propagated to the evaluator as
`EvalError::RuntimeError(...)`.

- **Wrong arity.** Anything other than 2 arguments returns
  `RuntimeError::ArityMismatch`.
- **Non-vector `values`.** "values must be a rank-1 vector,
  got rank N".
- **Non-scalar `k`.** "k must be a scalar".
- **Non-integer / negative `k`.** "k must be a non-negative
  integer, got <value>".
- **`k > len(values)`.** "k = K exceeds the vector length
  N".
- **`k == 0`.** Returns an empty vector shaped `[0]`. Not an
  error.

## What this contract does NOT cover

- Masked-logits sampling. Use `top_k(logits, k)` +
  `sample(logits, temperature, seed)` for that pathway.
- Multi-axis top-k. Saga 20 ships the rank-1 case only;
  callers that need per-row top-k can slice and loop.
- Stable determinism across endian / float modes. We rely on
  `f64::partial_cmp`, which matches IEEE-754 ordering.
