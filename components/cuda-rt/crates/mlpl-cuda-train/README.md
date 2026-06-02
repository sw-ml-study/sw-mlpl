# mlpl-cuda-train

CUDA on-device training primitives for sw-MLPL -- the candle
"approach A" analog of `mlpl-mlx-train` (framework autodiff +
optimizer, zero hand-written backward formulas). Linux + x86_64 +
the `cuda` feature gate everything; off-target the crate is empty.

## Step 001 spike result (GO)

candle's CUDA backend builds and runs on this dev host:

- GPU: NVIDIA GeForce RTX 5060 Ti (Blackwell, `sm_120`, 16 GB)
- Toolkit: CUDA 13.2 (`/opt/cuda`), driver 595.71.05
- Versions (pinned in `Cargo.lock`): candle-core 0.9.2,
  candle-nn 0.9.2, cudarc 0.19.7

Three gated tests pass on the GPU (`tests/spike_tests.rs`): CUDA
device init, candle autodiff gradient matching the closed-form
least-squares gradient, and Adam collapsing the loss on-device.

## Build / test incantation

cudarc/`bindgen_cuda` find the toolkit via the CUDA env vars, and
`CUDA_COMPUTE_CAP=120` targets Blackwell:

```bash
export PATH="/opt/cuda/bin:$PATH"
export CUDA_ROOT=/opt/cuda CUDA_PATH=/opt/cuda
export CUDA_COMPUTE_CAP=120
export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"

cargo test -p mlpl-cuda-train --features cuda -- --test-threads=1
```

Off-target / without the feature the crate exports nothing and is a
no-op, so the rest of the workspace stays cross-platform-buildable:

```bash
cargo clippy --all-targets -- -D warnings   # stub, passes anywhere
```
