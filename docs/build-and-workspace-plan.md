# Build modes + workspace restructure plan

## Three build modes

The project produces two artifacts: the **WASM pages** (the web playground,
built by trunk into `pages/`) and the **`mlpl-serve` binary** (the connect
server, optionally GPU-aware). What you can build depends on the host:

| Host | Server | Pages | Script |
|------|--------|-------|--------|
| Linux x86_64 + NVIDIA GPU | `mlpl-serve --features cuda` | yes | `scripts/build-cuda.sh` |
| Apple Silicon (Darwin/arm64) | `mlpl-serve --features mlx` | yes | `scripts/build-mlx.sh` |
| Anything else (no GPU) | CPU-only or none | yes | `scripts/build-pages.sh` |

- The **pages are identical** across hosts -- they are pure WASM and have
  no GPU code; GPU work always runs server-side via `?connect=`.
- `scripts/build-cuda.sh` / `build-mlx.sh` build the matching GPU server
  AND the pages, and refuse to run on the wrong platform (pointing at the
  right script). `mlx-rs` only builds on macOS and the CUDA candle stack
  only links on Linux, so cross-building a GPU server is not possible.
- `scripts/serve-cuda.sh` (run, not build) starts the CUDA connect server.
  An `serve-mlx.sh` analog should be added for Apple.

### Verifying MLX after CUDA work

The CUDA saga added `cuda-model` / `cuda-rt` and the `grad_optim_cuda*`
fast paths, all triple-gated (`feature = "cuda"`, `target_os = "linux"`,
`target_arch = "x86_64"`). The MLX paths (`grad_optim_mlx*`,
`feature = "mlx"`, macos/aarch64) are untouched and the demos were updated
in parallel. BUT MLX cannot be compiled off-Apple, so after any change
touching `mlpl-eval` / `mlpl-serve` / the demos, run on an Apple Silicon
Mac: `scripts/build-mlx.sh` then exercise the MLX LoRA demo via
`scripts/serve-mlx.sh` + `?connect=`.

## Workspace restructure (future saga)

Today all 48 components share one `target/` via `.cargo/config.toml`, and
GPU-specific crates are gated by feature/target rather than separated into
workspaces. That works but bundles five compilation contexts (native
interpreter, async server, WASM target, CUDA stack, MLX stack) in one
tree, so a Linux box compiles MLX path-deps it can never link and vice
versa, and `target/` churns across contexts.

Proposed grouping into top-level cargo workspaces, each with its own
`target/` (trading whole-repo ergonomics for build isolation):

1. **shared / common / WASM** -- core, array, parser, runtime, eval (the
   device-agnostic parts), viz, web-*, wasm, monitoring, web-demos. Builds
   on any host; produces the pages. The bulk of the crates.
2. **cuda** -- `cuda-model`, `cuda-rt`, `mlpl-cuda-train`, the
   `grad_optim_cuda*` glue, and a `mlpl-serve` built with `--features
   cuda`. Linux x86_64 only.
3. **mlx** -- `mlx-model`, `mlpl-mlx-train`, the `grad_optim_mlx*` glue,
   the vendored `mlx-rs`, and a `mlpl-serve` built with `--features mlx`.
   Apple Silicon only.

### Concrete seam design (worked out 2026-06-11)

The device-trait seam (commit 282b0870) was step one. To actually MOVE the
impls to sibling crates, the constraint discovered is: the architecture
RECOGNIZERS are interpreter-coupled and must stay in `mlpl-eval`:
- `grad_optim_mlx_demo::extract_xy` calls `crate::eval::eval_expr` to
  evaluate the X/Y argument expressions.
- `demo_layout` / the MLP recognizer read `ModelSpec`s and bindings.

So the split is recognition-in-eval, compute-in-gpu-crate:

1. `mlpl-eval::eval_adam` runs the recognizers (it has the interpreter)
   and obtains `(layout, X, Y)`.
2. It calls a reshaped `GpuAdamStep`:
   `run_lora_step(&self, layout: &DemoLayout, x: &DenseArray, y:
   &DenseArray, hp: &AdamHp, env: &mut dyn GpuEnv)` (and `run_mlp_step`).
3. `GpuEnv` is a NARROW accessor trait (keeps mlpl-eval's general surface
   tight -- no broad `pub` on get/set): just what the compute needs --
   read/write a named adapter weight, and read/write an Adam moment
   buffer `(opt, param, suffix) -> DenseArray`. `Environment` impls it.
4. `DemoLayout` / `MlpLayout` + `AdamHp` + `GpuAdamStep` + `GpuEnv` become
   the public seam (in `mlpl-eval`, cfg "any GPU").

Then the cuda/mlx crates contain ONLY candle/mlx compute (build the
device tensors from the accessor, forward/backward via candle/mlx
autograd, adam update, write back via the accessor) -- no interpreter, no
ModelSpec parsing. The cycle breaks because `mlpl-eval` no longer
constructs the impls; `mlpl-serve` (feature-gated) installs the right
`GpuAdamStep` via `Environment::install_gpu_step` at session creation.

Recommended staging (each verifiable; MLX needs an Apple build):
- S1 [DONE, c15799d7]: introduce `GpuEnv` + reshape `GpuAdamStep` to
  `run_*_step(layout, x, y, hp, &mut dyn GpuEnv)`; move recognition into
  `eval_adam`. Impls stay in mlpl-eval. CUDA fast path verified unchanged
  (device("cuda") train100 0.49s); MLX verified on Apple (938 passed).
  Trait/impl split into gpu_env.rs + env_gpu.rs (facade discipline).
- S2 [DONE, ab895c82]: break the cycle. `Environment::new` reads a
  process-global registry (gpu_registry) instead of naming the concrete
  impl; binaries (mlpl-serve run_main, mlpl-repl run) call
  `register_default_gpu_step()` at startup. A cfg-gated in-crate fallback
  to `default_gpu_step()` keeps the in-crate GPU tests green until S3.
  (Deviation: process-global register-at-startup, not a per-session
  `install_gpu_step` -- Environment is built at ~6 sites.)
- S3 [next]: create `mlpl-cuda-eval`, move the cuda compute
  (grad_optim_cuda*). Make the seam pub (GpuAdamStep, GpuEnv, AdamHp) +
  DemoLayout/LoraNames fields pub so the sibling crate can read them.
  Delete the in-crate fallback; binary registers `mlpl_cuda_eval`'s step.
  Move the cuda demo tests to the new crate. Verify on Linux.
- S4: create `mlpl-mlx-eval`, mirror. Verify on Apple.

Other tensions:
- `mlpl-serve` either lives in shared (optional cuda/mlx deps per
  workspace) or is built per-GPU-workspace.
- Separate `target/`s cost disk; keep the shared workspace's the big one.

Until the restructure lands, the three build scripts above are the
supported way to build each mode.
