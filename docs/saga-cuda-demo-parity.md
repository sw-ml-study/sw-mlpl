# Saga: CUDA demo parity

## Vision

Bring the remaining MLX demos to CUDA, so every connect/GPU demo has a
CUDA equivalent runnable on a Linux + NVIDIA peer. `cuda-foundation`
built the engine + the first CUDA demo (LoRA fine-tune); this saga adds
the rest, starting with the one the user asked for: **CUDA tic-tac-toe
fine-tune**.

## What `cuda-foundation` already gives us

- The CUDA engine: `mlpl-cuda-rt` / `-elementwise` / `-nn`, the
  traceable `mlpl-cuda-forward` primitives + `lora_linear`,
  `mlpl-cuda-train` (candle `loss_and_grads` + `adam_update`), and the
  `device("cuda")` dispatch + `eval_adam` CUDA branch (head-only-LoRA
  fast path `grad_optim_cuda`).
- `mlpl-serve --features cuda` connect peer + `/v1/devices` gating.

## The tic-tac-toe gap (first deliverable)

The MLX tic-tac-toe demo uses a DIFFERENT architecture than the LoRA
demo: the board-policy **MLP** -- `Chain[LinearLora, relu, LinearLora]`,
with BOTH linears LoRA-adapted (4 adapters: A1,B1,A2,B2). On the MLX
side that's a separate fast path (`grad_optim_mlx_mlp` +
`grad_optim_mlx_mlp_step` + `mlpl-mlx-model::mlp_forward`). cuda-foundation
ported only the head-only-LoRA path, so the CUDA MLP path is new work.

## Steps

1. **cuda-mlp-model** -- add `mlp_forward` + `MlpWeights` to
   `mlpl-cuda-model` (the candle analog of `mlpl-mlx-model::mlp.rs`):
   `h = relu(lora_linear(x,W1,A1,B1,s1) + b1)`,
   `logits = lora_linear(h,W2,A2,B2,s2) + b2`, mean softmax
   cross-entropy. Gated test trains the 4 adapters via candle AdamW and
   the loss drops -- on the GPU.
2. **cuda-mlp-dispatch** -- `grad_optim_cuda_mlp` + `grad_optim_cuda_mlp_step`
   in `mlpl-eval` (analogs of the MLX MLP glue): recognize
   `Chain[LinearLora, relu, LinearLora]`, extract the 4 adapters + frozen
   bases, run `loss_and_grads` over `mlp_forward`, adam-step the adapters
   (moments in `env.optim_state`). Wire into `eval_adam`'s CUDA branch
   (try the LoRA-head path, then the MLP path). Parity test: a
   `device("cuda")` MLP LoRA loss curve matches the CPU path within fp32
   tol, on the GPU.
3. **cuda-tictactoe-demo** -- the demo (mirror the MLX tic-tac-toe:
   MLPL engine -> self-play dataset -> LoRA-fine-tune the board policy
   under `device("cuda")` -> before/after win-rate). `demos/` script +
   org page (published on the GPU) + a `"CUDA tic-tac-toe fine-tune"`
   registry entry (`{requires_connect, device: Cuda}` + catalog +
   literate map). Rebuild `pages/`; verify runnable on the connected
   CUDA peer.

## Out of scope (later in this saga or a follow-on)

- CUDA `tiny_lm`, `neural_thicket`, and `cuda_remote` equivalents (the
  rest of the demo matrix in `docs/gpu-demos-roadmap.md`). Add as
  further steps once tic-tac-toe lands, or split to a follow-on.

## References

- `components/mlx-model/crates/mlpl-mlx-model/src/mlp.rs` (the MLX MLP
  forward to mirror) + `mlp_tests.rs`.
- `mlpl-eval/src/grad_optim_mlx_mlp.rs` + `grad_optim_mlx_mlp_step.rs`
  (the MLX MLP eval glue to mirror).
- `docs/saga-cuda-foundation.md` (the engine this builds on).
