# Future saga: general GPU training (remove the CPU fall-back)

## Problem

`device("cuda")` / `device("mlx")` do **not** accelerate arbitrary models.
`eval_adam` (`components/eval/crates/mlpl-eval/src/grad_optim.rs:216-225`)
tries exactly two hand-written model shapes per backend:

- `grad_optim_cuda::try_lora_adam` / `grad_optim_mlx::try_lora_adam` --
  head-only LoRA.
- `grad_optim_cuda_mlp::try_mlp_adam` / `grad_optim_mlx_mlp_step` --
  the tic-tac-toe board-policy MLP.

Each calls a **bespoke** candle/mlx forward (`demo_forward` in
`mlpl-cuda-model` / the MLX analog), differentiated by the backend's
autograd. For ANY other model -- notably the full transformer base
(`embed -> rms_norm -> causal_attention -> residual -> rms_norm ->
linear -> cross_entropy`) -- both return `None` and `eval_adam` falls
through to the **CPU autograd tape**. The tape uses CPU `DenseArray` ops
directly; it does NOT route through the per-op GPU dispatch
(`device_dispatch.rs::dispatched_call`).

Consequences, measured on a 12-core Linux box + RTX 5060 Ti (2026-06-10):

- `device("cuda")` base-model pretrain ran **19.3s at 0% GPU** -- it
  *silently* fell back to CPU. No warning.
- The LoRA fine-tune (the one supported shape) hit **62% GPU**.
- So the "CUDA demo" was mostly slow CPU; the GPU was barely used.

## Quick interim win (do first, small)

`device("cuda") { ... }` silently runs on CPU when unsupported -- users
think CUDA is engaged when it is not. Emit a one-time **warning / metric**
when a `device("<gpu>")` block (or its `adam`) falls back to CPU, so the
REPL surfaces "this step ran on CPU (GPU fast path not available for this
model)". Cheap; directly fixes the "are we even using CUDA?" confusion.
Hook point: the `None`/`None` fall-through in `eval_adam`, and/or
`dispatched_call` when `device != cpu` but no GPU op matched.

## Goal

`device("cuda")` / `device("mlx")` trains ANY model on the GPU; no silent
CPU fall-back. The live telemetry sparkline then shows real GPU load for
base-model training, and `loss_metric` (an extra forward pass) also runs
on-device instead of dragging the loop onto the CPU.

## Two approaches

**A. Device-aware autograd tape (general).** Give the tape GPU-resident
arrays and route its forward + backward ops through `dispatched_call`
when `env.device()` is a GPU. Pro: works for *any* model, reuses the
existing per-op dispatch surface (matmul/add/cross_entropy/... already
exist for CUDA + MLX). Con: needs a GPU-resident array that stays on
device across tape ops (today `DenseArray` is host memory; naive
dispatch would round-trip host<->device every op and be slower than CPU).
This is the right long-term architecture.

**B. Per-architecture candle/mlx forward (fast, narrow).** Extend the
`demo_forward` approach to the full transformer block in
`mlpl-cuda-model` + the MLX analog, differentiated by candle/mlx
autograd. Pro: one fused GPU graph, fast. Con: hand-ported per
architecture; does not generalize to arbitrary user models.

Recommendation: **A** for the general fix; B only if a specific
architecture needs peak throughput.

## Sketch of steps (approach A)

1. GPU-resident array type (or a device tag on `DenseArray`) whose op
   results stay on device; host<->device transfer only at boundaries.
2. Tape forward routes ops via `dispatched_call` when device is a GPU,
   keeping intermediates on device.
3. Tape backward (grad) ops on device.
4. `adam` update on device tensors; optimizer moments resident on device.
5. Parity tests: base-transformer train step GPU vs CPU within fp32 tol.
6. Retire the bespoke `demo_forward` fast paths (or keep as optimizations).

## Demo note

Until this lands, the CUDA/MLX LoRA demos deliberately **skip the
CPU-only base pretrain** and train the LoRA adapters directly on the GPU
(`train 1500`, rank-16), so they are fast and GPU-bound (~8s @ ~60% GPU).
`last_losses` fills from the training loss automatically -- no
`loss_metric` (which would force an extra CPU forward pass per step). See
`mlpl-web-demos-basic/src/lm.rs` (CUDA_LORA_FINETUNE / MLX_LORA_FINETUNE).
