# Engram support in sw-MLPL -- saga plan (MLX first, CUDA later)

Source analysis: `docs/engram-support-in-sw-mlpl.txt` (the E1-E10
sketch), mapped onto the repo as it stands after the 2026-07 eval
decomposition (Environment + capability traits below the hub, the
Model DSL cluster in `mlpl-eval-models`, GPU step seam in
`mlpl-eval-state`, MLX compute in `mlpl-mlx-eval`/`mlpl-mlx-rt`).

## What the repo already gives us

- `ModelSpec` (mlpl-eval-core) is the layer enum an `Engram`
  variant slots into; the freshly-split `mlpl-eval-models` crate is
  where apply/lowering lives.
- Tokenizers (byte-level + BPE), embedding/attention/RMSNorm
  layers, autograd tape, adam, `train` blocks, `device("mlx")`
  scoping, the gpu-step registry seam, and the demo/e2e harness.
- The MLX runtime today round-trips every op through CPU
  `DenseArray` (f64) -- the doc's "immediate technical priority"
  (persistent device tensors) is REAL and is its own saga.

## Constraint found during mapping: integer ops

MLPL arrays are f64. Multiply/add/mod hashing is EXACT in f64 up
to 2^53, so a first Engram can hash on the existing array type
with bit-for-bit CPU/MLX parity. The paper's XOR mixing is NOT
f64-representable -- faithful XOR needs an integer tensor path
(MLX has native int arrays; the CPU side would need a dtype or a
u64 sidecar). This is decision D4 below.

## Proposed sagas (each one agentrail saga)

| Saga | Content | Deliverable |
| --- | --- | --- |
| E1 engram-primitives | shift/pad, rolling n-gram hash (f64-safe mul-add-mod per D4), flattened multi-head gather, head-offset tables; CPU reference + parity fixtures | `ngram_hash(...)` builtin + Engram Hash demo (CPU==MLX indices gate) |
| E2 engram-dsl | `engram(...)` ModelSpec variant + EngramSpec validation/introspection (`:describe e` with parameter/byte accounting), `apply_engram(e, h, ids)` (per D3), gate mode "concat" first, serialization schema | Learnable Phrase Memory demo (train tables standalone) |
| E3 engram-tiny-lm | Engram inside selected Tiny-LM blocks (AfterAttention hook), frozen-base training, gate/collision stats (`engram_stats`), baseline-vs-engram comparison | Tiny LM + Engram demo with stats panel |
| E4 mlx-persistent-tensors | THE runtime redesign: TensorHandle {Cpu, Mlx}, device-resident values across expressions, explicit sync, device-resident grads/optimizer state | Tiny-LM train on MLX faster than CPU (currently slower) |
| E5 engram-mlx (COMPLETE 2026-08-03) | resident selection-matmul gather (exact scatter-ADD backward; device gather rejected -- no scatter-add in vendored mlx-rs), dev concat/split, parity suites (bit-exact hashing, 0.000000 trajectory drift), flat 19/28/332/1 seam profile, crossover ~d=128 | Tiny LM Engram on MLX demo (web + native); benchmarks.md E5 section |
| E6 sparse-rows | IndexedRows parameter update (rows+values gradients), sparse SGD/Adam for tables | 100M-scale table training in budget |
| E7 checkpoint-import | safetensors + HF tokenizer JSON + config normalization (DecoderModelSpec), Llama-family importer, quantized linear repr | `load_transformer(...)` |
| E8 retrofit-small | inject_engram/freeze/unfreeze on a 100M-1B import | small retrofit demo |
| E9 twelve-b-mlx | quantized 12B+ load + Engram inject + generate on Apple | the headline demo |
| E10 engram-cuda | (follow-on, per user) Candle-CUDA first behind the same EngramBackend contract, cudarc kernels after profiling | CUDA parity |

Crate placement: new `components/engram/` with `mlpl-engram-core`
(spec/hash/metadata/backend contract, no tensor deps) and the
backend impls folded into the existing mlx/cuda component pattern;
demos ride the existing demos.toml pipeline with a new "Engram"
dropdown group.

## Decision points (user guidance requested)

- D1 DECIDED (2026-07-30): Engram starts now; the
  eval-decomposition saga is PAUSED with its remaining steps
  (inspect-out, device-out, grad-out, models-crate subdivision,
  spine-tidy) documented in docs/eval-env-design.md +
  docs/eval-decomposition-saga.md for a resume saga.
- D2 DECIDED: CPU-demos-first -- E1-E3 before the E4 MLX
  persistent-tensor runtime redesign.
- D3 DECIDED: Option A, explicit `apply_engram(e, h, ids)`; the
  internal trait keeps a LayerContext door open.
- D4 DECIDED: f64-exact mul-add-mod hashing on the existing array
  type (bit-identical CPU/MLX parity gates); XOR-faithful integer
  mixing becomes an upgrade when integer tensors land with E4.

Each saga gets its own agentrail plan at start.
