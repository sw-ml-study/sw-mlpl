# `estimate_train` Contract (Saga 22 step 001)

## Purpose

`estimate_train(model, steps, batch_size, seq_len [,
dtype_bytes]) -> [5]` returns a feasibility /
resource-estimation snapshot for a planned training
run. The output is honest-approximate at ~2x accuracy
and lets users on limited hardware gate real work with
`feasible(...)` (step 002) or just read the numbers
and decide by eye. Pure-math over the `ModelSpec` tree;
no actual work is performed.

## Signature

```
estimate_train(model, steps, batch_size, seq_len)
estimate_train(model, steps, batch_size, seq_len, dtype_bytes)
```

- `model` -- bare ident bound in `env.models` or
  expression evaluating to `Value::Model`. Same arg
  shape as `clone_model` / `freeze` / `lora` /
  `embed_table`.
- `steps`, `batch_size`, `seq_len` -- positive scalar
  numbers. Non-positive or non-finite values error.
- `dtype_bytes` (optional, default `8` = f64) --
  positive scalar number. Supply `4` for an f32
  what-if, `2` for f16/bf16. MLPL runs f64 today, so
  the override only affects the estimate, not any
  real allocation.
- Returns rank-1 `[5]` f64 in fixed order:
  ```
  [params, vram_bytes, disk_bytes, flops, wall_seconds]
  ```
  Labels pinned in this contract. No keyword-arg
  sugar; `params = result[0]`, etc.

## Derivations

Let `P` = set of parameter names walked from the
ModelSpec tree. Let `|P|` = sum of sizes of every
array bound to a name in `P`. Let `T = P \ frozen`
(parameters not in `env.frozen_params`). Let `depth`
= count of parameterized nodes (Linear, LinearLora,
Embedding, Attention). Let `hidden` = widest
`d_model` / `out_dim` / `in_dim` observed across
those nodes.

- **params** = `|P|`. Frozen params count here.
- **vram_bytes** =
  `|P| * dtype_bytes                 // weights`
  `+ |T| * dtype_bytes                // gradient`
  `+ 2 * |T| * dtype_bytes            // Adam m + v`
  `+ batch * seq * hidden * depth * dtype_bytes * 4`
- **disk_bytes** = `|P| * dtype_bytes` (one full
  checkpoint; save-frequency multiplier deferred).
- **flops per step**, summed over the tree:
  - `Linear`: `2 * in_dim * out_dim * batch`
  - `LinearLora`: `2 * in * out * batch + 2 * in *
    rank * batch + 2 * rank * out * batch`
  - `Embedding`: `2 * batch * vocab * d_model`
  - `Attention`: `8 * d_model^2 * batch * seq + 4 *
    seq^2 * d_model * batch` (four d_model x d_model
    projections + QK^T + AV)
  - `Chain(children)`: sum of child flops
  - `Residual(inner)`: inner's flops (wrapper is
    free)
  - `Activation / RmsNorm`: 0 (sub-dominant)
- **flops total** = `flops_per_step * steps`.
- **wall_seconds** = `flops_total / (gflops * 1e9)`,
  where `gflops` is parsed from
  `env.get_string("mlpl_device_throughput_gflops")`
  when set, else the conservative default **50
  GFLOPS** (laptop CPU lower bound). Saga 22 step
  002's `calibrate_device()` writes this key based
  on an observed matmul benchmark.

## Assumptions + accuracy target

- f64 default because MLPL runs f64 today. The
  `dtype_bytes` override lets you ask "what if I
  switched to f16?" -- the estimator halves the
  memory terms cleanly.
- `activation_factor = 4` is a conservative safety
  multiplier. Activation memory is the most
  architecture-sensitive component; this factor
  covers Q/K/V/softmax temporaries + residuals
  + KV cache in a simple product rather than a
  per-node walk. Expect ~2x accuracy overall --
  fine for "will this fit" decisions, too
  imprecise for "exactly how long will this take".
- FLOPS ignore softmax, layer norm, elementwise
  ops, and residual wrappers (all sub-dominant
  relative to matmul for any nontrivial model).
  Also ignores reductions, optimizer math, and
  dataloader overhead.
- Wall-clock is linearly scaled by FLOPS /
  throughput. Real throughput varies 2x+ with
  matmul dimensions and op mix; this is why the
  number is explicitly approximate.

## Determinism

Bit-identical outputs for two calls with the same
`(model, steps, batch, seq, dtype_bytes)` in the
same `env`. No PRNG. Different `env` state (frozen
set, throughput string) changes the output.

## Error cases

All errors surface as `EvalError::Unsupported(...)`
or `EvalError::BadArity(...)`.

- **Wrong arity.** 4 or 5 args required; anything
  else -> `BadArity`.
- **Non-model first arg.** "estimate_train: first
  argument must be a model".
- **Non-scalar numeric args.** "estimate_train:
  <name> must be a scalar, got rank N".
- **Non-positive numeric args.** "estimate_train:
  <name> must be positive, got <v>" (catches
  zero, negative, NaN, infinity).
- **Model with no trainable parameters.** e.g.,
  `chain(relu_layer(), tanh_layer())`. "estimate_
  train: model has no trainable parameters".

## What this contract does NOT cover

- **Inference-only sizing.** A future `estimate_
  infer` variant would drop grad + Adam terms from
  the VRAM formula. Deferred until a concrete use
  case surfaces; for now, ballpark it as VRAM /
  4 (weights-only, no activation allowances).
- **Dynamic profiling / exact measurement.** This is
  an estimator, not a profiler. Real measurement
  lives in a future saga's `profile { ... }` scope.
- **Distributed / multi-GPU estimation.** Single-
  device. Saga 17 (CUDA + distributed) owns the
  multi-device story.
- **HuggingFace download cost.** No weights are
  fetched. Saga 22 step 002's `hypothetical_model`
  provides structural ModelSpecs for SmolLM / Llama
  / Qwen without download; the actual download +
  safetensors pipeline is a separate future saga.
- **Save-frequency / checkpoint multipliers.**
  `disk_bytes` is one full checkpoint. Runs that
  save every N steps should multiply manually.
- **Automatic shrinking / recovery.** The estimator
  reports; it does not rewrite the program. The
  Saga 22 step 002 `feasible(est, budget)`
  guard pattern is the intended usage.
- **Activation-checkpointing / gradient
  accumulation.** Not modeled; they would
  dramatically change the activation term and are
  a separate knob.
