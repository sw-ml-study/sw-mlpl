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
- `StrList { items: Vec<String> }` -- Saga 29 step 002: list of
  strings. Produced by `[...]` literals whose elements all
  evaluate to `Value::Str`. Sibling to `Array` (the numeric
  `DenseArray` path) -- the same surface syntax dispatches by
  element kind. Empty `[]` keeps the back-compat numeric
  (`DenseArray`) shape; mixed-kind elements raise
  `MixedArrayLitElements`. Accessed via `list_len(xs)` for
  length; indexing / iteration are out of scope until a
  follow-up.

Field access on a record returns the inner value (which may itself
be any variant including another `Record`). Field access on any
other Value variant errors with `FieldOnNonRecord`. Unknown field
errors with `FieldNotFound { requested, available }` so the user
gets the list of valid keys.

`[...]` array literals dispatch on the kinds of their evaluated
elements: all `Value::Str` -> `StrList`; all `Value::Array` (the
existing numeric path) -> a stacked `DenseArray`; mixed kinds ->
`MixedArrayLitElements { kinds }`. Empty `[]` continues to
produce an empty `DenseArray` for back-compat.

### Image-tensor builtins (Saga 29 step 003)

`load_images(dir, [H, W])` (native-only, gated on the `image-io`
Cargo feature) reads every PNG / JPEG file under `dir` (resolved
inside the sandbox root set by `Environment::set_data_dir`),
decodes via magic-byte dispatch to the `png` or `jpeg-decoder`
crate, bilinear-resizes to `(H, W)`, normalizes u8 RGB to f64
in `[-1, 1]` via `v / 127.5 - 1.0`, and stacks the results into
a `Value::Array` of shape `[N, 3, H, W]` with axis labels
`[batch, channel, y, x]`. The WASM build (no `image-io`
feature) raises `EvalError::Unsupported` pointing at
`load_preloaded("pets_tiny")` instead. The MLX peer wire does
not encode images; live decode happens on the native side and
the resulting tensor is what crosses the wire.

`load_preloaded("pets_tiny")` returns a `Value::Record` with
three fields: `X` (a `Value::Array` of shape `[200, 3, 64, 64]`
with the same axis labels as live `load_images`), `Y` (a
`Value::Array` of shape `[200]` with `batch` axis label;
`0 = cat`, `1 = dog`), and `names` (a `Value::StrList` of the
200 source filenames, e.g. `["Abyssinian_1.jpg", ...]`). The
fixture is built offline from the Oxford-IIIT Pet dataset by
`cargo run --example build_pets_tiny --features image-io
-p mlpl-eval` and shipped as `crates/mlpl-eval/data/pets_tiny.bin`
via `include_bytes!`, so the WASM REPL has the fixture
available without any live decode.

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
  "device-tensor", "string-list".
- `MixedArrayLitElements { kinds }` -- Saga 29 step 002: `[...]`
  literal contained more than one kind of element (e.g. mixing
  strings and numbers). `kinds` is the per-position list of
  `value_kind()` results in source order so the tutoring
  message can show which element broke the rule.

## What This Contract Does NOT Cover

- Parsing (that is `mlpl-parser`)
- Array storage internals (that is `mlpl-array`)
- Built-in function implementations (that is `mlpl-runtime`)
- Trace recording (that is `mlpl-trace`)
- Compilation or JIT
