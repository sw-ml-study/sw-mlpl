# `scatter` Contract (Saga 20 step 003)

## Purpose

`scatter(buffer, index, value)` returns a copy of `buffer`
with the single rank-1 entry at `index` replaced by `value`.
Pairs naturally with loops (`repeat N { ... }`,
`for i in list { ... }`) that produce one number per
iteration and need to accumulate them into a rank-1 vector.

Used by the Neural Thickets workflow to build a
`losses : [N]` vector inside a perturbation sweep, one
iteration at a time.

## Signature

```
scatter(buffer, index, value) -> buffer'
```

- `buffer` -- rank-1 float vector.
- `index` -- scalar integer in `[0, len(buffer))`.
- `value` -- scalar float.
- Returns a new rank-1 vector of the same length as `buffer`,
  equal to `buffer` elementwise except at `index`, which
  holds `value`.

## Semantics: "return new array"

At the MLPL source level, `scatter` is a functional
operation: `scatter(b, i, v)` does not mutate `b`. Users
thread the result back through a re-assignment
(`b = scatter(b, i, v)`) to accumulate over time. A test
pins this behaviour: the source binding's data is unchanged
after a `scatter` call on it. The runtime is free to
optimise the copy internally; the observable semantics are
unchanged.

## Naming

`scatter` is distinct from `scatter_labeled` (the
multi-class 2D visualisation helper). Name collisions are
avoided because the evaluator checks `scatter_labeled`
exactly in its analysis-helper dispatch and falls through to
this runtime builtin for the bare `scatter` name.

## Error cases

All errors are `RuntimeError::InvalidArgument { func:
"scatter", reason }`, propagated as
`EvalError::RuntimeError(...)`.

- **Wrong arity.** Anything other than 3 arguments returns
  `RuntimeError::ArityMismatch`.
- **Non-vector `buffer`.** "buffer must be a rank-1 vector,
  got rank N".
- **Non-scalar `index` or `value`.** "index must be a
  scalar" / "value must be a scalar".
- **Non-integer or negative `index`.** "index must be a
  non-negative integer, got <value>".
- **`index >= len(buffer)`.** "index I out of bounds for
  buffer of length N".

## What this contract does NOT cover

- Multi-axis scatter / gather. Saga 20 ships the rank-1
  scalar-write case only. Rank-2+ or vectorised scatter is
  a follow-up.
- Accumulation semantics (scatter-add). This builtin
  overwrites; it does not add. Callers that need
  accumulation can read-modify-write:
  `scatter(b, i, b[i] + delta)` if / when indexing lands.
- Device dispatch. `scatter` is CPU-only in Saga 20. The
  Neural Thickets demo uses it to build a scalar-per-variant
  losses vector outside any `device("mlx") { }` block.
