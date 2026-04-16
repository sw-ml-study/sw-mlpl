# Using MLPL with CUDA

> **Status:** planned -- Saga 17 (`v0.14` target), *after* Saga
> 14's MLX backend (`v0.11`). Not yet shipped. Treat everything
> below as design, not reference.

## Why CUDA

After Apple Silicon, the next hardware target for any serious ML
workload is NVIDIA. Saga 17 is the CUDA backend plus distributed
execution hooks (multi-GPU, multi-node). It lands *after* MLX
(Saga 14) because:

- MLX's lazy-graph + automatic-fusion design is closer to MLPL's
  existing runtime shape, so it teaches us what a second backend
  needs from the shared abstractions.
- Apple Silicon is the baseline dev hardware for this project.
  Landing MLX first means dev laptops are on the accelerated
  path before CUDA requires a remote box or a workstation.

A program that trains on MLX (Saga 14) should train on CUDA
(Saga 17) without source changes. Both backends share the MLPL
surface: Model DSL, `apply`, `grad`, optimizers, labels.

## Saga 17 shape

`docs/plan.md` sketches Saga 17 as:

1. **`mlpl-cuda` runtime target.** Sibling to `mlpl-rt` /
   `mlpl-mlx`. Re-exports CUDA-backed versions of the primitive
   set. A feature flag on the `mlpl` facade selects the backend.
2. **`device("cuda")` scoped form.** Same language-level shape
   as `device("mlx")`, so code is portable between the two
   accelerators by changing one string.
3. **Distributed execution.** Multi-GPU on one node first
   (`device("cuda:0")` / `device("cuda:1")`), then multi-node
   via NCCL. The orchestration layer lives above the runtime
   target -- think all-reduce inside `adam` for data-parallel
   training.

## Intended API (not shipped)

```mlpl
# Place on the first CUDA device
X : [batch, feat] = device("cuda:0") { randn(7, [1024, 64]) }

# Same Model DSL, same training loop
mdl = chain(linear(64, 256, 0), relu_layer(), linear(256, 64, 1))
train 100 {
  adam(cross_entropy(apply(mdl, X), targets), mdl,
       0.01, 0.9, 0.999, 0.00000001);
  loss_metric = cross_entropy(apply(mdl, X), targets)
}

# Data-parallel across two GPUs (Saga 17 Phase 3)
distribute "data_parallel" [cuda:0, cuda:1] {
  train 100 { ... }
}
```

`distribute` is deliberately a new scoped form rather than a
per-op flag so the compiler knows where collective operations
(all-reduce, broadcast) should land. Shape and label propagation
across the `distribute` boundary is part of the saga design.

## Fine-tuning use case

Full fine-tuning (not just LoRA) on CUDA is the primary workload
Saga 17 targets. The workflow will look like:

1. Load a pretrained checkpoint (format shipped alongside
   Saga 15's LoRA / quantization work).
2. Place the model on one or more CUDA devices.
3. Stream training batches through `apply` +
   `cross_entropy` + `adam` inside a `train { }` loop, exactly
   like the CPU Saga 13 Tiny LM demo -- just with larger models
   and longer runs.
4. Checkpoint back to disk; resume or serve the result.

LoRA / QLoRA / quantization (Saga 15) ship before Saga 17, so
adapter-only fine-tuning on CUDA is available by the time CUDA
lands.

## What you can do today

- **Write a model with the Model DSL.** It'll port to CUDA
  unchanged once Saga 17 ships. Use labeled axes so the shapes
  are self-documenting -- label mismatches surface at compile
  time in the `mlpl!` macro and run time in the interpreter.
- **Run on the `mlpl build` CPU path.** Today's near-native Rust
  is within a small factor of what CUDA will deliver on
  tiny models (see `docs/benchmarks.md`); the CUDA win shows up
  only when the workload fills the device. Measuring the CPU
  baseline now means the Saga 17 release has a clear speedup
  story.
- **Track `docs/plan.md` / `docs/saga.md`.** Saga 17 is blocked
  on Saga 14. If Saga 14 is still pending, CUDA is at least two
  sagas out.

## Current non-goals

- **Custom CUDA kernels.** The backend uses cuBLAS / cuDNN and a
  small hand-written kernel set for the ops those libraries don't
  cover. An MLPL user writing custom `__device__` code is out of
  scope.
- **ROCm / Intel Gaudi / TPU.** Not planned. The architecture
  (swappable runtime target crate behind a feature flag) would
  admit them eventually, but nobody's signed up to port.
- **cuBLAS handle management exposed to MLPL.** The backend owns
  handles, streams, and events; the MLPL surface stays at the
  array-op level.

## Why *not* just use PyTorch?

You can. MLPL's value proposition versus "write your model in
PyTorch":

- **One source, many backends.** The same `.mlpl` code runs on
  CPU (interpreter, `mlpl build` native binary, WASM) today, and
  on MLX then CUDA as those backends land. Under PyTorch you
  maintain separate code (or at least separate device-placement
  trees) for the same logic.
- **Statically checkable shapes and labels.** PyTorch's named
  tensors are opt-in and don't propagate through every op; in
  MLPL they're the default once you use annotation syntax.
- **A deployment story without Python.** `mlpl build` produces a
  self-contained native binary with no interpreter, no Python,
  no dynamic code loading (see `docs/compiling-mlpl.md`). PyTorch
  deployments ship the interpreter.

When Saga 17 ships, `docs/compiler-guide.md` will grow a section
on the `mlpl-cuda` feature flag, and the bench harness
(`docs/benchmarks.md`) will get a CUDA column. Until then: CPU-
first, MLX next, CUDA after.

## Related

- `docs/plan.md` -- Saga 17 plan (CUDA + distributed)
- `docs/saga.md` -- live saga status
- `docs/using-mlx.md` -- the first accelerator backend (Saga 14)
- `docs/benchmarks.md` -- where CUDA numbers will land once the
  backend ships
