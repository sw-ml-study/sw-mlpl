# Eval Contract

## Purpose

Define the behavioral spec for expression evaluation in MLPL.
`mlpl-eval` takes a parsed AST and produces values (arrays, scalars)
by interpreting it. It depends on core, parser, array, and runtime.

## Key Types and Concepts

### Value

The result of evaluating an expression.

- Wraps `DenseArray<f64>` from `mlpl-array` (scalars are rank-0 arrays)
- May later include other types (strings, booleans, functions)

### Environment

A name-to-value mapping for variable bindings.

- Supports let-binding and lookup
- Scoped (nested environments for future function calls)

### Evaluator

Walks the AST and produces values.

- `evaluate(ast, env) -> Result<Value, EvalError>`
- Dispatches on AST node kind: literal, binop, function call, etc.
- Calls into `mlpl-array` for array construction and operations
- Calls into `mlpl-runtime` for built-in function dispatch

## Invariants

- Evaluation is deterministic (same AST + env -> same result)
- Type mismatches produce explicit errors, not panics
- Every evaluation step can be traced (future trace integration)

## Error Cases

- `EvalError` is local to `mlpl-eval`
- `UndefinedVariable(Identifier)` -- name not in environment
- `TypeMismatch { expected, got }` -- wrong value kind for operation
- `ArityMismatch { expected, got }` -- wrong argument count
- `ArrayError(mlpl_array::ArrayError)` -- propagated from array ops

## What This Contract Does NOT Cover

- Parsing (that is `mlpl-parser`)
- Array storage internals (that is `mlpl-array`)
- Built-in function implementations (that is `mlpl-runtime`)
- Trace recording (that is `mlpl-trace`)
- Compilation or JIT
