# `clone_model` Contract (Saga 20 step 001)

## Purpose

`clone_model(m) -> Model` deep-copies a model spec and its
parameters so the caller can mutate the copy (via
`perturb_params`, `adam`, manual `env.set`, etc.) without
affecting the source. Foundation builtin for the Neural
Thickets workflow: one base model plus N independent
perturbation variants.

## Signature

```
clone_model(m) -> Model
```

- `m` -- a model identifier (bare `Expr::Ident` bound in the
  environment's model registry) **or** any expression that
  evaluates to `Value::Model` (e.g. `linear(...)`, `chain(...)`).
- Returns `Value::Model` wrapping a new `ModelSpec` whose
  parameter names are disjoint from those of the source.

## Fresh-name scheme

Each parameterised node allocates one fresh id via
`env.next_model_id` (the same counter `linear`, `embed`, and
`attention` use at construction time), then formats names with
the existing convention:

- `Linear` -> `__linear_W_{id}`, `__linear_b_{id}`
- `Embedding` -> `__embed_E_{id}`
- `Attention` -> `__attn_Wq_{id}`, `__attn_Wk_{id}`,
  `__attn_Wv_{id}`, `__attn_Wo_{id}`

Parameter-free nodes (`Activation`, `RmsNorm`, `Chain`,
`Residual`) are copied structurally and recursively without
allocating new names.

## Behavioural guarantees

1. **Fresh, disjoint names.** The clone's parameter names do
   not collide with the source's, with any other clone's, or
   with any previously allocated model's. Calling
   `clone_model` N times produces N fully disjoint name sets.
2. **Bit-identical forward output (pre-perturbation).** Before
   any mutation, `apply(clone, X)` produces the same array as
   `apply(source, X)` for every valid `X`. Values are copied
   by `DenseArray` clone.
3. **Mutation isolation.** Assigning a new tensor to a clone's
   parameter (via `env.set`, `perturb_params`, `adam`, etc.)
   does not change the source's parameter values.
4. **Trainable registration.** Every new parameter name is
   added to the trainable-params set (`env.is_param(new_name)`
   returns `true`), so optimizers and `grad` treat the clone's
   params exactly like any constructed model's.
5. **Device propagation.** Each new parameter inherits the
   source parameter's device tag
   (`env.tensor_device(old_name)`). A clone of a model that
   was moved to MLX via `to_device(base, "mlx")` is itself
   MLX-tagged, so `apply(clone, X)` inside a `device("mlx")
   { ... }` block does not trigger a device-mismatch error.

## Error cases

- **Wrong arity.** `clone_model(a, b)` or `clone_model()`
  returns `EvalError::BadArity { func: "clone_model",
  expected: 1, got }`.
- **Non-model argument, bare ident path.** `clone_model(x)`
  where `x` is not a model returns
  `EvalError::Unsupported("clone_model: 'x' is not a model")`.
- **Non-model argument, expression path.** Any expression that
  evaluates to `Value::Array` or any other non-model value
  returns
  `EvalError::Unsupported("clone_model: argument must evaluate
  to a model")`.
- **Missing param value in environment.** If a referenced
  parameter name is not bound, returns
  `EvalError::UndefinedVariable(old_name)`. This should not
  happen in practice -- model constructors always bind their
  params -- but is surfaced rather than silently ignored.

## What this contract does NOT cover

- `perturb_params` (step 002) -- uses `clone_model` output as
  its input; specified in its own contract.
- Deep copies of non-model values (arrays, tokenizers, etc.).
  MLPL has `=` for that.
- `Compile-to-Rust` / `mlpl-rt` parity. The compiled path does
  not need `clone_model` for the Saga 20 demo; port if future
  parity tests require it.
- Session / environment-level cloning. Cloning an
  `Environment` wholesale is not a language-level operation.
