# `estimate_hypothetical` Contract (Saga 22 step 002)

## Purpose

`estimate_hypothetical(name, steps, batch_size,
seq_len [, dtype_bytes, lora_rank]) -> [5]` returns
the same `[params, vram_bytes, disk_bytes, flops,
wall_seconds]` shape as `estimate_train`, computed
from a **hardcoded spec table** for a named open-
weights model. Answers "how big would SmolLM-135M +
LoRA fine-tune be on my laptop?" WITHOUT downloading
weights or materializing parameter arrays.

## Why a separate builtin

`estimate_train` walks a real `ModelSpec` + reads
parameter shapes from `env`. For a hypothetical model
we would need to materialize gigabytes of zero arrays
just to ask the question. `estimate_hypothetical`
skips the spec-tree + env dance entirely and computes
directly from the per-model dimension table.

## Signature

```
estimate_hypothetical(name, steps, batch_size, seq_len)
estimate_hypothetical(name, steps, batch_size, seq_len, dtype_bytes)
estimate_hypothetical(name, steps, batch_size, seq_len, dtype_bytes, lora_rank)
```

- `name` -- string literal selecting from the
  hardcoded table (below).
- `steps`, `batch_size`, `seq_len` -- positive
  scalar numbers.
- `dtype_bytes` (optional, default `8` = f64) --
  what-if element size.
- `lora_rank` (optional, default `0` = full
  fine-tune) -- if > 0, treat every Linear
  (attention projections + FFN + head) as a LoRA-
  wrapped layer with this rank, freezing the base.
  Reduces grad + Adam memory proportionally.
- Returns rank-1 `[5]` f64 in the same layout as
  `estimate_train`, so `feasible(est, budget)`
  composes directly.

## Supported model names

| Name | vocab | d_model | layers | intermediate |
|------|-------|---------|--------|--------------|
| `smollm-135m` | 49152 | 576 | 30 | 1536 |
| `smollm-360m` | 49152 | 960 | 32 | 2560 |
| `smollm-1.7b` | 49152 | 2048 | 24 | 8192 |
| `llama-3.2-1b` | 128256 | 2048 | 16 | 8192 |
| `qwen-2.5-0.5b` | 151936 | 896 | 24 | 4864 |

Dims from each model's published config. The table
approximates total parameter count within ~30% of
each model's reported size -- close enough for
"will this fit" decisions, too coarse for exact
sizing.

## Math

Let `V = vocab, D = d_model, L = layers, F =
intermediate, dt = dtype_bytes, r = lora_rank, B =
batch, S = seq, N = steps`.

```
per_layer_params = 4 * D * D            # Q, K, V, O
                 + 2 * D * F            # FFN up + down
params = V * D                         # embedding
       + L * per_layer_params
       + D * V                         # output head
trainable = params                     # when r = 0
         or L * (4 * (D + D) * r        # attention LoRA adapters
              + 2 * (D + F) * r)       # FFN LoRA adapters
            + (D + V) * r              # head LoRA adapter
depth = 2 + 2 * L
vram = (params + trainable + 2 * trainable) * dt
     + B * S * D * depth * dt * 4      # 4x activation factor
disk = params * dt
per_step = 2 * B * V * D                         # embedding gather
         + L * (8 * D * D * B * S                # QKVO projections
              + 4 * S * S * D * B                # QK^T + AV
              + 4 * D * F * B * S)               # FFN matmuls (fwd)
         + 2 * B * D * V                         # output head
flops = per_step * N
wall  = flops / (gflops * 1e9)
```

`gflops` is read from
`env.get_string("mlpl_device_throughput_gflops")`;
default 50 GFLOPS if unset. Saga 22's
`calibrate_device()` writes that key.

## What this contract does NOT cover

- **Auto-download weights.** This is a structural
  estimator; the actual HuggingFace download +
  safetensors load is a separate future saga.
- **Per-architecture tuning.** We assume all models
  are prenorm decoder-only Transformers with 4
  attention projections + 2 FFN linears per layer.
  Mistral-7B fits; Llama-3.2-1B fits; GPT-2 fits;
  mixture-of-experts models would need a new entry.
- **Exact parameter counts.** The table is within
  ~30% of each model's reported size. Good enough
  for "will this fit" gating; use it as a floor.
- **GQA / MQA kv-head sharing.** Counted as full
  MHA for simplicity; a future enhancement could
  add a `kv_heads` field.
- **Tokenizer + dataset memory.** Estimates are for
  the model only; add your own overhead for
  dataloader / tokenizer tables.
- **User-defined models.** Name must match the
  table exactly; unknown names error helpfully.
