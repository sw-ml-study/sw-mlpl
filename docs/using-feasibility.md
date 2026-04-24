# Using MLPL for Feasibility Checking + Resource Estimation

> **Status:** reference. Shipped in Saga 22 (v0.15.0).

## What this is about

On a limited-hardware PC (laptop CPU, 4-8 GB of RAM,
a modest GPU or none), the risky thing is not writing
the code -- it is committing hours of wall-clock (or
worse, gigabytes of disk, or a dead GPU OOM) to a
run that was doomed from the start. Saga 22 ships
four builtins that answer the four questions a
limited-PC user actually cares about BEFORE the
first optimizer step:

- **How many parameters?** (easy from the ModelSpec)
- **How much VRAM to train?** (weights + grad +
  Adam moments + activation)
- **How much disk per checkpoint?** (weights x
  dtype)
- **How long will it take?** (FLOPS / device
  throughput)

Accuracy target is ~2x. Activation memory is
architecture-sensitive; the 4x safety factor covers
Q/K/V/softmax temporaries + residuals. FLOPS ignore
softmax, layer norm, and elementwise ops (all
sub-dominant to matmul for any real model). Wall-
clock depends on how accurately the device's matmul
throughput is calibrated. Good enough for "will this
fit" decisions; too imprecise for "exactly how long
will this take".

## The builtins

### `estimate_train(model, steps, batch, seq [, dtype_bytes]) -> [5]`

Pure-math walk over a real `ModelSpec`. Returns a
rank-1 `[5]` f64 array:

```
[params, vram_bytes, disk_bytes, flops, wall_seconds]
```

Contract:
[`contracts/eval-contract/estimate.md`](../contracts/eval-contract/estimate.md).

Frozen parameters (e.g., LoRA base after `lora(m,
rank, alpha, seed)`) count in `params` + `disk` +
the forward-memory slice of `vram` but NOT in the
grad + Adam memory terms. That is why a LoRA fine-
tune has dramatically lower VRAM than a full fine-
tune of the same base.

Default `dtype_bytes = 8` (f64, what MLPL uses).
Pass `4` or `2` for f32 or f16/bf16 what-ifs; the
memory terms scale cleanly but MLPL does not
actually support those dtypes yet.

### `calibrate_device([size]) -> gflops`

Benchmark matmul throughput on the active device,
cache the result into
`env.mlpl_device_throughput_gflops`. Every subsequent
`estimate_train` / `estimate_hypothetical` reads
that key, so wall-clock numbers go from "50 GFLOPS
default floor" to "the honest observed number".

Contract:
[`contracts/eval-contract/calibrate-device.md`](../contracts/eval-contract/calibrate-device.md).

Default size = 1024; pass a smaller `size` for a
quick calibration in tests or demos. Runs through
the same `device::dispatched_call` path as every
other matmul, so
`device("mlx") { calibrate_device() }` measures MLX
throughput. No device suffix on the key; re-
calibrate when you switch devices.

### `estimate_hypothetical(name, steps, batch, seq [, dtype_bytes, lora_rank]) -> [5]`

Same `[5]` output as `estimate_train`, but computed
from a hardcoded spec table for a named open-weights
model. Answers "how big would SmolLM-135M + LoRA
fine-tune be on my laptop?" WITHOUT needing to
download weights or materialize parameter arrays --
the purely-structural dimensions live in the table.

Contract:
[`contracts/eval-contract/estimate-hypothetical.md`](../contracts/eval-contract/estimate-hypothetical.md).

Supported names today:

| Name | vocab | d_model | layers | intermediate |
|------|-------|---------|--------|--------------|
| `smollm-135m` | 49152 | 576 | 30 | 1536 |
| `smollm-360m` | 49152 | 960 | 32 | 2560 |
| `smollm-1.7b` | 49152 | 2048 | 24 | 8192 |
| `llama-3.2-1b` | 128256 | 2048 | 16 | 8192 |
| `qwen-2.5-0.5b` | 151936 | 896 | 24 | 4864 |

`lora_rank > 0` computes the LoRA-wrapped variant:
frozen base, small adapter set, dramatically reduced
grad + Adam memory.

### `feasible(estimate_result, budget) -> 0/1`

The guard pattern. Takes a `[5]` estimator output
and a `[3]` budget `[vram, disk, wall]`; returns
`1.0` if every non-zero budget is satisfied, `0.0`
otherwise. Zeros in the budget mean "skip this
dimension".

Contract:
[`contracts/eval-contract/feasible.md`](../contracts/eval-contract/feasible.md).

## The SmolLM-LoRA worked example

Classic limited-PC question: "Can I fine-tune
SmolLM-135M on my 8-GB-RAM laptop?"

```mlpl
# What would a full fine-tune cost?
full = estimate_hypothetical("smollm-135m", 100, 4, 512)

# What about a LoRA fine-tune at rank=8?
lora = estimate_hypothetical("smollm-135m", 100, 4, 512, 8, 8)

# Budget: 4 GB VRAM, 10 GB disk, 10 minutes wall.
budget = [4000000000.0, 10000000000.0, 600.0]

feasible(full, budget)    # likely 0.0 -- full FT will not fit
feasible(lora, budget)    # much more likely 1.0
```

The only way to know which side of the line you are
on is to run the numbers, and the only way to run
the numbers is with the estimator. The tradeoff is
clear: LoRA keeps base disk but drops almost all
grad + Adam memory, which on a 135M-param model is
the difference between "laptop-feasible" and "nope".

## The guard pattern

```mlpl
m = chain(embed(V, d, 0), transformer_block, head)
est = estimate_train(m, 1000, 32, 512)
if feasible(est, [vram_budget, disk_budget, wall_budget]) {
  train 1000 { adam(cross_entropy(apply(m, X), Y), m, 0.001, 0.9, 0.999, 0.00000001) }
} else {
  :describe est
  "aborting: est exceeds budget"
}
```

The point is NOT that the estimator is magic. It is
that the estimator is cheap: walking the spec tree
costs microseconds, calibrate_device costs a few
seconds, and you get to spend minutes thinking
instead of hours regretting a misspecified
`batch_size`.

## The demo

`demos/feasibility.mlpl` walks the pipeline end to
end (CPU, CLI):

```bash
./target/release/mlpl-repl -f demos/feasibility.mlpl
```

Estimates a tiny toy model, calibrates the device,
re-estimates (wall drops an order of magnitude),
shows `estimate_hypothetical` on SmolLM-135M both
full and LoRA, and demonstrates the `feasible(...)`
gate.

## What is NOT shipped (deferred follow-ups)

- **HuggingFace Hub download + safetensors loading.**
  Separate future saga. `estimate_hypothetical`
  answers "how big would this be" without needing
  to download.
- **f16 / bf16 tensor dtype machinery.** MLPL runs
  f64 today. The `dtype_bytes` override in the
  estimators lets you ask about hypothetical half-
  precision runs, but the actual half-precision
  tensor path is a future saga.
- **Dynamic profiling / exact measurement.** The
  estimator is honest-approximate. Real per-op
  profiling needs a `profile { ... }` scope and is
  out of Saga 22's scope.
- **Distributed / multi-GPU estimation.** Single-
  device only. Saga 17 (CUDA + distributed) owns
  the multi-device story.
- **Activation checkpointing / gradient
  accumulation.** Not modeled. Both change the
  activation term dramatically and need explicit
  knobs.
- **Tokenizer / dataset memory overhead.**
  Estimates cover the model only. Add your own
  overhead for the dataloader + tokenizer tables.
- **Soft warnings.** `feasible` is boolean pass /
  fail. A future variant could return per-dimension
  headroom so the user sees "you are within 2x of
  VRAM and 10x of wall-clock".
- **Automatic shrinking / recovery.** The estimator
  reports; it does not rewrite the program.

## Related

- `contracts/eval-contract/estimate.md`
- `contracts/eval-contract/calibrate-device.md`
- `contracts/eval-contract/estimate-hypothetical.md`
- `contracts/eval-contract/feasible.md`
- `demos/feasibility.mlpl`
- Sibling docs: `docs/using-lora.md` (Saga 15),
  `docs/using-embeddings.md` (Saga 16).
