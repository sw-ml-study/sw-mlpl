# Saga: CUDA foundation (Linux GPU backend)

## Vision

Bring real NVIDIA/CUDA GPU acceleration to sw-MLPL on Linux,
mirroring the now-complete MLX (Apple) track. The end state: a
CUDA-equivalent demo + literate `.org` doc for each MLX demo, and a
UI that connects in `?connect=` mode to an `mlpl-serve` peer running
on a Mac (MLX) AND/OR on an Arch Linux box (CUDA), gating demos by
the device the connected peer actually offers.

This first saga is the **foundation + one demo end-to-end**, proven
on real hardware. A follow-up saga mirrors the remaining MLX demos.

## Why now, why here

This dev host is an **RTX 5060 Ti (Blackwell, sm_120, 16 GB), CUDA
13.2 at `/opt/cuda`, x86_64-linux-gnu, 1.6 TB free**. Unlike the
MLX track -- which this Linux box can never build or test -- a CUDA
track is fully buildable and TDD-testable in-session with real GPU
parity tests, not Apple-only glue.

## Foundation already in place (CUDA-aware plumbing)

- **`Device::Cuda`** exists in the web-demos registry
  (`mlpl-web-demos-types`), rendered visible-but-not-runnable on the
  public live demo.
- **`Value::DeviceTensor { device: String, .. }`** already names
  `"cuda"` as a future device (`mlpl-eval-types`).
- **`device("cuda") { .. }`** already parses and pushes onto
  `Environment::device_stack`; it currently falls back to CPU with a
  one-time warning because no CUDA dispatch exists.
- **Connect mode (SSE)** routes eval to any `mlpl-serve` peer; the
  `peer_dispatcher` abstraction dispatches a whole `device(...)`
  block server-side.

## Backend decision (the analog of MLX step 005's fork)

**Chosen: `candle`** (HuggingFace `candle-core` autograd via
`Tensor::backward()` + `candle-nn` optimizers, on `cudarc`). This
mirrors the MLX track's winning "approach A": use the framework's
built-in autodiff + optimizer, zero hand-written backward formulas.
Pure Rust, no multi-GB libtorch download. The load-bearing risk is
Blackwell sm_120 / CUDA 13.2 support in `cudarc`/`candle` -- the
step-1 spike de-risks exactly that before any wider work.

Rejected: `tch` (libtorch -- heavy + Blackwell/CUDA-13 build
freshness risk) and `cudarc`-only (= MLX "approach B", ~hand-written
kernels + autodiff, explicitly rejected on the MLX track).

## Crate plan (mirror the MLX crates 1:1)

| MLX crate | Role | CUDA crate to build |
|---|---|---|
| `mlpl-mlx-rt` | forward ops | `mlpl-cuda-rt` (matmul/shape/convert) + `mlpl-cuda-elementwise` (arith) + `mlpl-cuda-nn` (activations/reductions) |
| `mlpl-mlx-forward` | embed / rms_norm / attention / cross-entropy | `mlpl-cuda-forward` |
| `mlpl-mlx-train` | autodiff + on-device Adam | `mlpl-cuda-train` |
| `mlpl-mlx-model` | `demo_forward` assembly | `mlpl-cuda-model` |
| `mlpl-mlx-serve` | server-side peer dispatcher | `mlpl-cuda-serve` |

All CUDA crates triple-gate `cfg(all(feature = "cuda", target_os =
"linux", target_arch = "x86_64"))`; absent any gate, dispatch is a
stub returning `None` and `device("cuda")` falls back to CPU with a
one-time warning (mirror `mlpl-eval/src/device.rs`).

## Steps

1. **cuda-backend-spike** -- DONE (GO). Gated crate
   `components/cuda-rt/crates/mlpl-cuda-train` proves candle's CUDA
   backend on the RTX 5060 Ti: 3 GPU tests pass (device init,
   autodiff gradient == closed-form least-squares gradient, Adam
   collapses the loss on-device). Pinned candle-core/nn 0.9.2 +
   cudarc 0.19.7 build for `sm_120` / CUDA 13.2; build incantation
   in the crate README (`CUDA_COMPUTE_CAP=120`, `/opt/cuda` env).
   candle (approach A) confirmed viable -- the rest of the saga
   proceeds on it.
2. **mlpl-cuda-rt** -- DONE. The core tensor ops split into two
   warning-free sibling crates (the eager surface is too large for
   one crate under the <=4 fn/module + <=4 module/crate budgets):
   `mlpl-cuda-rt` (convert/device plumbing, `matmul`, `reshape`,
   `transpose`) and `mlpl-cuda-elementwise` (`add`/`sub`/`mul`/`div`
   with scalar broadcast, `neg`). Both gated; 6 GPU parity tests
   pass vs the CPU path within fp32 tol. The nn surface
   (activations, softmax/reduce, `cross_entropy`) is split out to
   step 2b (`mlpl-cuda-nn`): candle lacks a `prod` reduction and
   `cross_entropy` parity is fiddly, and those ops overlap the
   step-4 candle forward.
2b. **mlpl-cuda-nn** -- DONE. The activations turned out to be
   elementwise unary maps, so they moved into `mlpl-cuda-elementwise`
   (`exp`/`log`/`relu`/`sigmoid`/`tanh` alongside `neg`, via a shared
   `unary!` macro). `mlpl-cuda-nn` is then cleanly reductions
   (`mean`/`argmax` on the GPU; `reduce_mul` delegates to the CPU --
   candle has no `prod`), normalization (`softmax`/`log_softmax`), and
   `cross_entropy` (GPU row-LSE + CPU gather, mirroring `mlpl-rt`).
   3 logic modules (reduce/norm/loss), all warning-free; parity-tested
   vs the CPU path on the GPU. Baseline held (0 new failures/warnings).
3. **cuda-dispatch-wire** -- generalize `try_mlx_dispatch` /
   `dispatched_call` in `mlpl-eval/src/device.rs` so `device("cuda")`
   routes through `mlpl-cuda-rt`; CPU fallback + one-time warning
   preserved; device dispatch tests extended for cuda.
4. **cuda-forward-and-model** -- `mlpl-cuda-forward` (embed,
   rms_norm, causal_attention, cross_entropy) + `mlpl-cuda-model`
   (`demo_forward`), each parity-tested vs the CPU primitives and
   the MLX reference.
5. **cuda-lora-demo** -- `mlpl-cuda-train` (candle autodiff +
   on-device Adam) wired into `eval_adam`'s CUDA branch (analog of
   `grad_optim_mlx.rs`); first CUDA demo `demos/lora_finetune_cuda.mlpl`
   + literate `examples/literate/cuda-lora-finetune.org`; flip the
   registry entry to runnable-on-connected-CUDA-peer. Parity: the
   `device("cuda")` LoRA loss curve matches the CPU path within fp32
   tol.
6. **cuda-connect-peer** -- `mlpl-cuda-serve` so
   `mlpl-serve --features cuda` on Linux is a connect-mode peer;
   add a `GET /api/devices` capability probe so the UI knows whether
   a connected server offers `mlx`, `cuda`, or neither, and gates
   demos by the live peer instead of a static guess. This is what
   makes "connect to a Mac and/or this Arch box" honest.

## Out of scope (this saga)

- Mirroring the other 3 MLX demos (tiny_lm, neural_thicket,
  mlx_remote) + their org docs -- a FOLLOW-UP saga once the vertical
  slice is green.
- SmolLM2 loader / LLaMA-style ops (saga `local-gpu-agentic`
  Phases 4-5, now CUDA candidates -- later).
- Agentic `:ask` Phase 2/3 -- parked on `main` as
  `local-gpu-agentic` step 011, platform-agnostic, pick up on
  either host.

## References

- `docs/saga-local-gpu-agentic.md` -- the MLX track this mirrors;
  device tiers, "measurable learning" metric, demo-download policy.
- `mlpl-eval/src/device.rs` -- the single dispatch site to generalize.
- `components/native-rt/crates/mlpl-mlx-{rt,forward,train}`,
  `components/mlx-model/`, `components/serve/.../mlpl-mlx-serve` --
  the crates to mirror.
- candle: <https://github.com/huggingface/candle>;
  cudarc Blackwell/CUDA-13 support is the step-1 spike's job to pin.
