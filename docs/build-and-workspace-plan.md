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

Tensions to resolve in the saga:
- `mlpl-eval` currently owns BOTH `grad_optim_cuda*` and `grad_optim_mlx*`
  (feature-gated). To split cleanly, the GPU fast-paths would move behind
  a trait/registry the shared `eval` calls, with cuda/mlx crates providing
  the impls -- so `eval` itself stays device-agnostic and in the shared
  workspace. This is the same "device-aware" refactor that
  `docs/future-saga-gpu-training.md` needs, so do them together.
- `mlpl-serve` would either live in shared (with optional cuda/mlx
  dependencies pulled in per workspace) or be built per-GPU-workspace.
- Separate `target/`s cost disk; keep the shared workspace's `target/` as
  the big one and the gpu workspaces small.

Until the restructure lands, the three build scripts above are the
supported way to build each mode.
