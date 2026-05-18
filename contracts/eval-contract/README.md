# Eval Contract

## Purpose

Define the behavioral spec for expression evaluation in MLPL.
`mlpl-eval` takes a parsed AST and produces values (arrays, scalars)
by interpreting it. It depends on core, parser, array, and runtime.

## Key Types and Concepts

### Value

The result of evaluating an expression. Tagged union of:

- `Array(DenseArray)` -- the original numeric value (scalars are
  rank-0 arrays).
- `Str(String)` -- introduced for diagram type names, LLM prompts.
- `Model(ModelSpec)` -- callable Model DSL value (Saga 11).
- `Tokenizer(TokenizerSpec)` -- Saga 12 step 004.
- `BuiltinRef { name }` -- canonical first-class-ish op reference
  (Saga 21.5; `:foo` / `:max` / `:+` syntax).
- `DeviceTensor { peer, handle, shape, device }` -- peer-resident
  tensor (Saga R1 step 002). Strict-fault on cross-device CPU ops.
- `Record { fields: BTreeMap<String, Value> }` -- Saga 29 step 001:
  structured record literal value. BTreeMap-keyed for deterministic
  display + serialization + equality.

Field access on a record returns the inner value (which may itself
be any variant including another `Record`). Field access on any
other Value variant errors with `FieldOnNonRecord`. Unknown field
errors with `FieldNotFound { requested, available }` so the user
gets the list of valid keys.

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
- `FieldNotFound { requested, available }` -- Saga 29 step 001:
  record field lookup on a key the record does not have. The
  `available` list is sorted (BTreeMap key order) so the message
  is deterministic.
- `FieldOnNonRecord { receiver_kind, field }` -- Saga 29 step 001:
  field access on a non-record receiver. `receiver_kind` is one of
  "array", "string", "model", "tokenizer", "builtin-ref",
  "device-tensor".

## What This Contract Does NOT Cover

- Parsing (that is `mlpl-parser`)
- Array storage internals (that is `mlpl-array`)
- Built-in function implementations (that is `mlpl-runtime`)
- Trace recording (that is `mlpl-trace`)
- Compilation or JIT
