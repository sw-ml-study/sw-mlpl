# Error Handling in sw-MLPL

The user-facing guide: what exists today, the model behind it,
and what is queued. Design rationale and the APL-family survey
live in `docs/option-result-design.md`; the lens connection in
`docs/functional-lenses.md`. Every code line below was validated
against the live evaluator.

## The model: two planes, two bridges

sw-MLPL separates errors into two planes:

- **Control plane -- hard errors.** A mistake in program
  structure (wrong arity, shape mismatch, out-of-bounds index,
  `unwrap` of an `Err`) raises an `EvalError`: evaluation of the
  program stops and the REPL prints an `error:` line. Loud by
  design -- a teaching REPL must never silently absorb a bug.
- **Data plane -- error VALUES.** A fallible operation can
  instead RETURN a Result: `ok(v)` or `err(e)`. These are
  ordinary values; they print, bind, and flow like anything
  else. Nothing stops.

Two bridges connect the planes (both SHIPPED, spike step 011):

- `catch` DEMOTES a hard error into a value:
  `try { take(M, 0, 9) } catch e { fill([5], 0) }` -- an
  expression yielding body-or-handler; `e` is `{kind, message}`.
- `?` PROMOTES an `err` value into an early return:
  `v = r?; ...` inside a `def u:` body unwraps Ok or
  early-returns the Err (loud at top level). Spelled form:
  `check(expr)`.

## What works today

### Constructing Results

```text
r = ok(42)                       # Ok(42)
e = err("boom")                  # Err(boom)
e2 = err({kind: "index", message: "axis 3 out of range"})
```

`err`'s payload is ANY value. The RECORD payload is the
convention for structured errors (the "Error object"):
`kind` is a short machine tag, `message` the human string, and
extra context fields are welcome (`axis`, `rank`, ...).

### Consuming Results

```text
is_ok(r)                         # 1
is_err(e)                        # 1
unwrap(ok(42))                   # 42
unwrap(err("boom"))              # HARD error: UnwrapOnErr
unwrap_or(err("boom"), 7)        # 7
unwrap_or(ok([1,2,3]), 0)        # 1 2 3
err_message(e2)                  # {kind: index, message: axis 3 out of range}
if is_ok(r) { 100 } else { 200 } # 100
```

Guard rails are themselves loud: `unwrap` of an `Err` and any
accessor applied to a non-Result (`is_ok(5)` ->
`NotAResult`) are hard errors.

### Patterns

- **Default on failure**: `unwrap_or(fallible_thing, default)`.
- **Branch on outcome**: `if is_ok(r) { ... } else { ... }`.
- **Safe wrappers**: a `u:` function that bounds-checks before
  indexing and returns `ok(cell)` /
  `err({kind: "index", message: ...})` instead of exploding --
  see the safe-lens work in `docs/functional-lenses.md`.
- **Bulk data: use masks, not per-cell Results.** Validity over
  an array is a 0/1 mask (`eq`, `gt`, `batch_mask`), the
  array-native answer inherited from the APL family (q's typed
  nulls, APL fills). Reserve Results for control flow.

### Where you already meet these values

- `:upload <name>` binds `Ok({pixels, h, w})` or
  `Err("cancelled")`.
- Demo lines that fail print an `error:` entry and the demo
  continues to the next line -- hard errors are per-line in the
  web REPL transcript.

### The Option projections (shipped)

`get_value(r)` / `get_error(r)` -- Rust's `.ok()`/`.err()` with
the APL2 zilde flavor: each returns a 0-or-1-element vector
(`[]` when that side is absent), so `tally` is `is_some` and
`take(concat(get_value(r), [d], 0), 0, 0)` is `unwrap_or`.
Derived from the tag, so the illegal pair-states cannot arise.
Scalar payloads until Stage 6 `enclose`; non-scalar payloads
error with a message naming that gap. The Structure Zoo demo's
lens finale shows them against a safe deep-lens get.

### try/catch and `?` (shipped, spike step 011)

`try { body } catch e { handler }` catches HARD errors only,
binding `e` to `{kind, message}` (kinds are stable kebab-case
tags -- `shape`, `arity`, `runtime`, ... -- from
`error_kind`). `err(...)` values and break/continue/return
signals flow through untouched. `finally` stays deferred until
the language has a user-visible resource to clean up.

Postfix `expr?` desugars to `check(expr)`: Ok unwraps, Err
early-returns the whole Result from the enclosing `u:` function
via the return machinery; a stray top-level `?` on an Err is
loud (`UnwrapOnErr`). `u:` functions now accept Result, string,
and record arguments so pipelines compose.

## Queued (saga steps, in order)

1. **Error Handling demo** (step: error-handling-demo) -- a
   BASICS demo near Structure Zoo / Game of Life, four acts:
   the two planes side by side; the railway (safe gets,
   `unwrap_or`, projections); errors in USER-DEFINED functions
   (guard-then-ok/err as the way a `u:` body reports failure,
   plus a two-stage pipeline propagating the first Err); the
   two bridges (`catch` demoting an out-of-bounds `take`, `?`
   collapsing the manual propagation).
2. **Trap combinator** (needs first-class `u:` values) --
   `attempt(u:risky, u:handler)`, later sugar
   `u:risky :: u:handler`: the APL-family native form (J `::`,
   BQN `CATCH`, q `@[f;x;g]`). try/catch is the near-term surface;
   the adverse combinator is the point-free end state.

## Hard-error kinds you will meet (control plane)

| Kind | Trigger | Example message shape |
| --- | --- | --- |
| BadArity | wrong argument count | `rotate: expected 3, got 2` |
| ShapeMismatch | incompatible dims / OOB axis or index | `shape mismatch: 4 vs 2` |
| UnwrapOnErr | `unwrap(err(...))` | carries the err payload |
| NotAResult | Result accessor on a non-Result | names receiver kind + accessor |
| Unsupported | not valid in this context | free-form explanation |

When `try/catch` lands, `e` in the handler is the same
information reshaped into the canonical `{kind, message}`
record, so handlers can dispatch on `e.kind`.
