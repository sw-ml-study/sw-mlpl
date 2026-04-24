# `feasible` Contract (Saga 22 step 002)

## Purpose

`feasible(estimate_result, budget) -> 0/1` is the
guard pattern for feasibility checking. Takes a `[5]`
estimator output (from `estimate_train` or
`estimate_hypothetical`) and a `[3]` budget, returns
`1.0` if every non-zero budget is satisfied, `0.0`
otherwise.

Intended usage: gate a real training call on the
result.

```mlpl
est = estimate_train(model, 1000, 32, 512)
ok = feasible(est, [4_000_000_000, 10_000_000_000, 600])
# budget: 4 GB VRAM, 10 GB disk, 10 minutes
if ok { train 1000 { adam(...) } }
```

## Signature

```
feasible(estimate_result, budget) -> 0/1
```

- `estimate_result` -- rank-1 `[5]` f64, the layout
  produced by `estimate_train` and
  `estimate_hypothetical`: `[params, vram_bytes,
  disk_bytes, flops, wall_seconds]`.
- `budget` -- rank-1 `[3]` f64: `[vram_budget,
  disk_budget, wall_budget]`. A zero in any slot
  means "skip this dimension" (no check).
- Returns scalar f64: `1.0` if every budget with a
  non-zero value is satisfied
  (`estimate[dim] <= budget[dim]`), `0.0`
  otherwise.

## Zero-as-skip semantics

A `0.0` in any budget slot is interpreted as "I do
not care about this dimension". This lets callers
check one or two of the three without constructing a
separate guard:

```mlpl
# only check VRAM
feasible(est, [4_000_000_000, 0, 0])
# only check wall time
feasible(est, [0, 0, 3600])
```

This is a design choice that trades one magic value
for the cleaner two-arg `if feasible(...) { ... }`
pattern. `inf` or `-1` could have served the same
role; `0` is chosen because no legitimate budget is
ever zero.

## Error cases

All surface as `EvalError::Unsupported(...)` or
`EvalError::BadArity(...)`.

- **Wrong arity.** 2 args required.
- **`estimate_result` not rank-1 `[5]`**: `"feasible:
  estimate must be rank-1 [5], got <shape>"`.
- **`budget` not rank-1 `[3]`**: `"feasible: budget
  must be rank-1 [3] [vram, disk, wall], got
  <shape>"`.

## What this contract does NOT cover

- **Auto-recovery / shrinking.** `feasible` reports;
  it does not rewrite the program to fit. The user
  chooses between smaller batch / smaller model /
  LoRA / different device / nothing.
- **Soft warnings.** Any single budget over its
  limit fails; no "close to the edge" middle state.
  A future variant could return per-dimension
  pass/fail instead of a global boolean.
- **Runtime monitoring.** Check happens once at
  `feasible(...)` call time. The estimate does not
  update as training runs.
