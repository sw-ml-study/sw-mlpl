# `lora` Contract (Saga 15 step 002)

## Purpose

`lora(m, rank, alpha, seed)` wraps every `Linear` node in a
model's spec tree with trainable low-rank adapter matrices.
Clone-based: the returned model owns a fresh set of param
names that are disjoint from the source `m`'s names.

The LoRA decomposition: a frozen base `W: [in, out]` plus a
trainable low-rank delta `(alpha / rank) * A @ B` where
`A: [in, rank]` and `B: [rank, out]`. Forward (step 003):
`y = X @ W + (alpha / rank) * X @ A @ B + b`.

Step 002 ships the structural rewrite and the parameter
allocation only. Forward + autograd for `LinearLora` is step
003; `apply(lora_m, X)` returns a clear "not yet implemented
(Saga 15 step 003)" error until step 003 lands.

## Signature

```
lora(m, rank, alpha, seed) -> Model
```

- `m` -- model identifier (bare `Expr::Ident` bound in
  `env.models`) or any expression that evaluates to
  `Value::Model`.
- `rank` -- positive integer, `0 < rank <= min(in, out)` for
  every replaced `Linear`.
- `alpha` -- scalar f64; the forward uses `alpha / rank` as
  the scale applied to the adapter delta.
- `seed` -- scalar f64. The i-th replaced `Linear` uses PRNG
  seed `seed + i` when initializing its `A` adapter, so two
  `lora` calls with the same `seed` produce bit-identical A
  matrices (useful for reproducibility) while same-shape
  adapters within one call get independent deltas.
- Returns `Value::Model` wrapping a spec tree whose `Linear`
  nodes are now `LinearLora { w, b, a, b_adapter, in_dim,
  out_dim, rank, alpha }`. Non-`Linear` nodes
  (`Chain`, `Residual`, `Activation`, `RmsNorm`,
  `Embedding`, `Attention`) are cloned structurally and
  unchanged.

## Parameter names

Each `LinearLora` owns four parameters:

- `w: "__linear_W_{id}"` -- cloned base weight, shape
  `[in, out]`.
- `b: "__linear_b_{id}"` -- cloned base bias, shape
  `[1, out]`.
- `a: "__lora_A_{lid}"` -- adapter A, shape `[in, rank]`.
- `b_adapter: "__lora_B_{lid}"` -- adapter B, shape
  `[rank, out]`.

`ModelSpec::LinearLora`'s `params()` method returns all
four names. Callers that iterate a model's params
(optimizers, `:describe`, `collect_params` in grad.rs) see
the full set.

## Initialization

- `A <- randn(seed + i, [in, rank]) * (1 / sqrt(in))`. The
  running index `i` increments across replaced `Linear`s in
  tree-walk order; independent Gaussian draws per adapter.
- `B <- zeros([rank, out])`. The zero-init is the LoRA-
  standard "pre-training-step delta is zero" property: once
  step 003's forward lands, `apply(lora_m, X)` will match
  `apply(m, X)` elementwise before any gradient step.

Tests in step 002 pin that B is all zeros and that A has at
least one nonzero entry bounded by `10 * (1 / sqrt(in))` (a
generous practical ceiling on `randn * scale` per-element
magnitude; the 10x is the same rationale as the
`perturb_params` sigma bound).

## Automatic freeze of the base

After the rewrite, `lora` walks the student's full param
list and marks every name that is NOT a new adapter
(`__lora_A_*` or `__lora_B_*`) as frozen in
`env.frozen_params`. That includes:

- Cloned base `W`, `b` of every wrapped `Linear`.
- Embedding table (`__embed_E_*`) if the source has an
  `embed` node.
- Attention projections (`__attn_Wq_*`, `__attn_Wk_*`,
  `__attn_Wv_*`, `__attn_Wo_*`) if the source has an
  `attention` / `causal_attention` node.
- `rms_norm` nodes carry no parameters; no-op.

This matches the standard LoRA library convention --
"frozen base, trainable adapters" -- without the caller
having to enumerate which base params to freeze:

```mlpl
student = lora(base, 8, 16.0, 0)
train N { adam(loss, student, ...) ; ... }
# -> every base param untouched; only A, B train.
```

Callers who want full fine-tuning (train everything,
adapters AND base) call `unfreeze(student)` after `lora()`
to clear the frozen set for every student param.

## Device propagation

The adapter A, B inherit the cloned base `W`'s device tag
(`env.tensor_device`). If the source model was moved to MLX
via `to_device(base, "mlx")` before `lora()`, the student's
W, b, A, B all land on MLX.

## Error cases

- **Wrong arity** (not 4 args) ->
  `EvalError::BadArity { func: "lora", expected: 4, got }`.
- **Non-model first argument** ->
  `EvalError::Unsupported("lora: 'x' is not a model")` for a
  bare ident, or `"lora: first argument must evaluate to a
  model"` for an expression.
- **Non-scalar / negative / non-integer `rank`** ->
  `EvalError::Unsupported("lora: rank must be a non-negative
  integer, got ...")`.
- **`rank == 0`** ->
  `EvalError::Unsupported("lora: rank must be positive, got
  0")`.
- **`rank > min(in, out)`** for any replaced `Linear` ->
  `EvalError::Unsupported("lora: rank R exceeds min(in=IN,
  out=OUT) for this Linear")`. The message does NOT name the
  offending layer's structural path -- adding that is a
  quality-of-life follow-up.
- **Nested `lora()`** (applying `lora` to an already-lora'd
  model) ->
  `EvalError::Unsupported("lora: model already has LoRA
  adapters; nested lora() is not supported")`. Rationale: a
  user who wants to add more adapters on top of an existing
  LoRA can follow up with selective-attachment semantics
  that this saga defers.
- **Non-rank-2 base `W`** ->
  `EvalError::Unsupported("lora: base Linear W must be
  rank-2, got rank N")`. Defensive; in practice every
  Model-DSL-constructed `Linear` has a rank-2 W.

## What this contract does NOT cover

- **Forward + autograd for `LinearLora`.** Deferred to
  Saga 15 step 003. Until that ships, `apply(lora_m, X)`
  returns `EvalError::Unsupported("apply: LinearLora
  forward is not yet implemented (Saga 15 step 003)")` and
  the same for `grad(expr, lora_m)`.
- **Selective layer attachment.** Saga 15 ships the uniform
  "every `Linear` gets an adapter" variant. LoRA papers
  often only adapt attention projections (Wq, Wv); a
  `lora(m, rank, alpha, seed, layers: "attention_only")`
  follow-up would reuse Saga 20's family walker.
- **Adapter merging (`merge_lora`).** Baking the adapter
  back into the base W for inference deployment is a
  follow-up.
- **QLoRA / 4-bit quantization.** Quantization needs its
  own per-tensor scale/zero-point handling and parity
  harness; deferred to a future saga.
- **Multi-adapter composition / adapter routing.** One base
  with multiple swappable adapters, or adapter stacks, is a
  follow-up after plain LoRA proves out.
- **Nested `lora()`.** Explicit error today; re-enable once
  the semantics are nailed down.
