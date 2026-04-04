# Runtime Contract

## Purpose

Define the behavioral spec for built-in functions and operations in
MLPL. `mlpl-runtime` provides the implementations that `mlpl-eval`
dispatches to (reshape, transpose, arithmetic, reductions, etc.).
It depends on core and array.

## Key Types and Concepts

### Built-in Registry

A mapping from function names to implementations.

- `reshape`, `transpose`, `shape`, `rank` -- structural ops
- `add`, `sub`, `mul`, `div` -- element-wise arithmetic
- `sum`, `product`, `max`, `min` -- reductions (future)
- Registry is static for now; user-defined functions come later

### Function Signature

Each built-in has a fixed arity and expected argument types.

- Validated at call time by the evaluator
- Returns a `Value` or a `RuntimeError`

## Invariants

- Built-in names are stable once defined
- Arithmetic on arrays follows element-wise semantics
- Broadcasting rules (future) will be defined here, not in array

## Error Cases

- `RuntimeError` is local to `mlpl-runtime`
- `UnknownFunction(String)` -- name not in registry
- `InvalidArgument { func, reason }` -- arg fails precondition
- Propagates `ArrayError` from underlying array operations

## What This Contract Does NOT Cover

- Parsing or AST construction
- Array storage internals
- Trace recording
- User-defined functions (future)
- Broadcasting (future, but will live here when added)
