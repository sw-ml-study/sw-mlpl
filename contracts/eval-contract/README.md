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

`fetch_dataset(name)` (Saga 29 step 004, native-only via
`image-io`) is the live counterpart to `load_preloaded`.
For the v0.21 registry, only `name == "oxford_iiit_pet"` is
recognized: it downloads the upstream ~792 MB tarball via
`ureq` to `$MLPL_DATA_DIR/oxford-iiit-pet/images.tar.gz` on
first use, sha256-verifies against a pinned hash, untars to
`images/` if the dir isn't already populated, then runs the
same decode + bilinear-resize + normalize pipeline as
`load_images` at the demo's 128x128 resolution. Returns the
same `Value::Record { X, Y, names }` shape as `load_preloaded`,
but with `N = 7393` (the full Oxford-IIIT Pet count) instead
of 200. Pre-populated checkouts (existing tarball + extracted
`images/`) bypass HTTP entirely. Cat vs dog labels follow the
Oxford filename convention: uppercase prefix = cat (`0`),
lowercase prefix = dog (`1`).

Data-dir resolution for `fetch_dataset`: prefer the
`MLPL_DATA_DIR` environment variable; fall back to the
`Environment::data_dir` set by the terminal REPL's
`--data-dir` flag; without either, surface a tutoring error
(implicit 792 MB downloads are never on by default). The WASM
build raises `EvalError::Unsupported` pointing at
`load_preloaded("pets_tiny")`.

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

### ViT shape ops (Saga 29 step 005)

`patchify(x, P)` rearranges a `[B, C, H, W]` image batch into
`[B, N, P*P*C]` where `N = (H/P) * (W/P)`. `P` must divide both
`H` and `W`. Each row of the trailing axis is one patch
flattened in channel-outer order
(`out[b, n, c*P*P + dy*P + dx] = x[b, c, i*P+dy, j*P+dx]`
where `n = i*(W/P) + j`).

`concat(a, b[, axis])` has two arities. The 2-arg legacy form
(Saga 13) concatenates two rank-0 or rank-1 arrays into a 1-D
vector for generation loops. The 3-arg axis-aware form
(Saga 29 step 005) accepts any rank; both inputs must agree on
every dim except `axis` (sizes add). Initial release supports
`axis` in `{0, 1}`; higher axes error cleanly.

Both ops are differentiable on the autograd tape. `patchify`
adds `NodeKind::Patchify { parent, orig_shape, patch_size }`;
backward scatters the upstream `[B, N, P*P*C]` gradient back to
`[B, C, H, W]` image space (each output element comes from
exactly one input, so backward is the inverse indexing with no
accumulation). `concat` adds `NodeKind::Concat { left, right,
axis, left_size }`; backward splits the upstream gradient at
the seam and delivers each half to its parent.

### Batch-aware attention (Saga 29 step 008)

`apply(model, X)` and `attention_weights(model, X)` accept
both rank-2 `[seq, d_model]` and rank-3 `[B, T, d_model]`
input for single-head attention (`heads=1`). For rank-3 input
the forward path loops over the batch axis, runs the existing
rank-2 attention on each `[T, d_model]` entry, and stacks the
per-batch outputs back into `[B, T, d_model]`. The tape
lowering mirrors the structure by emitting `B` rank-2
attention chains and stitching their per-batch outputs via
`reshape(_, [1, T, d_model])` + chained `concat(_, _, 0)`,
which keeps every constituent op on the existing tape
without a new primitive.

Multi-head attention (`heads > 1`) still rejects rank-3 input
with a clean error pointing the user at `heads=1` or
explicit batch-fold; the multi-head + rank-3 path lands in
Saga 29 step 010 once the multi-head tape lowering is in
place. Causal masking works in both rank-2 and rank-3 paths
since the rank-3 path is just the rank-2 path applied B
times.

### Single-axis indexing (Saga 29 step 007)

`take(x, axis, idx)` drops one axis at a single integer index,
returning a `Value::Array` with rank `rank(x) - 1`. Per-axis
labels propagate: the dropped axis's label is removed and the
surviving labels are unchanged. Errors cleanly when `axis` is
out of range or `idx` is out of range for `dims[axis]`.

Differentiable on the tape via `NodeKind::Take { parent,
orig_shape, axis, idx }`. The backward scatters the upstream
gradient into a zero-filled array of `orig_shape`, placing the
upstream slice at position `axis = idx`. This is the
canonical "index_select"-style gradient: gradient flows only
through the picked slice.

Out of scope (followups): multi-index `gather`, slice ranges
`x[a..b]`, and negative indices.

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
