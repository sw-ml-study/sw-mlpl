# Saga E4: mlx-persistent-tensors (THE runtime redesign)

Per docs/engram-sagas-plan.md (E4) and the design brief in
docs/future-saga-gpu-training.md (Approach A: device-aware autograd
tape). Deliverable: Tiny-LM base-model training on MLX FASTER than
the CPU interpreter path (today it is 0.26x -- see
docs/benchmarks.md "Saga 14"). E5 (engram-mlx) builds on this.

State of the runtime (verified 2026-08-01):

- ONE numeric carrier: Value::Array(DenseArray), f64 host memory.
  Every MLX op round-trips f64->f32->f64 through
  mlpl-mlx-rt/src/common.rs (dense_to_mlx / mlx_to_dense_data) and
  forces .eval() per op, so MLX's lazy graph never fuses.
- The generic model path re-runs the whole forward a SECOND time
  (device.rs::materialize_tape_on_mlx) and still does backward on
  CPU. The gpu_step "fast path" (mlpl-mlx-eval) re-uploads every
  weight and every Adam moment from f64 host maps EVERY step.
- Metal is OFF: mlx-rs = "=0.25.3" with default-features = false,
  features = ["accelerate"] -- MLX today runs on Apple's CPU BLAS,
  not the GPU. The Metal toolchain IS available on this host
  (Xcode 26.6, xcrun finds metal), so the old blocker is gone.
- Value::DeviceTensor is TAKEN: it is the remote-peer handle
  (serve path), not a local device tensor. E4's handle needs a
  different name (TensorHandle / DeviceBuf below).
- Measured baselines (docs/benchmarks.md, warm, MLX/CPU):
  reshape_reduce 0.84x, tiny_lm_train_step 0.26x,
  neural_thicket_variant_loop 0.25x, lora_finetune_step 0.15x.
- wasm/Linux invariant: mlpl-array, mlpl-autograd, mlpl-eval-types,
  mlpl-eval-state are wasm-reachable and must never link mlx-rs.
  The handle must be an opaque trait object with the MLX impl
  registered behind the existing triple gate (same OnceLock
  inversion idiom as gpu_step / dispatch_hook).

Architecture decisions (proposed; flag disagreements early):

- A1: TensorHandle { Cpu(DenseArray), Dev(Arc<dyn DeviceArray>) }
  in a new small wasm-clean crate; DeviceArray is an opaque trait
  (shape, to_dense, as_any) and a DeviceOps registry supplies the
  resident op surface. mlpl-mlx-rt implements + registers it.
- A2: residency is introduced at the TAPE + OPTIMIZER level first
  (where the train-loop wins live), not by rewriting the ~200
  Value::Array call sites; language-surface reads sync via
  to_dense at the boundary (display, shape, metrics).
- A3: Metal ON (accelerate + metal): without the GPU, persistent
  tensors on CPU BLAS cannot beat the f64 CPU path.
- A4: precision contract unchanged in kind: CPU f64 stays the
  bit-exact reference; MLX path is f32 with the documented
  tolerance (FP32_TOL = 1e-4 per step; trajectory tolerance for
  multi-step loops), now applied to a path where state never
  returns to f64 between steps.

Steps (draft; each independently green + committable):

1. metal-on -- enable the metal feature next to accelerate in the
   four mlx crates (triple gate unchanged); verify the default MLX
   device is GPU at runtime; rerun cargo bench -p mlpl-bench
   --features mlx --bench mlx_vs_cpu and record the Metal-only
   delta as a new baseline table in docs/benchmarks.md. Isolates
   the toolchain risk before any type surgery.
2. tensor-handle-core -- the new handle crate: TensorHandle,
   DeviceArray trait, DeviceOps registry (upload, download, the
   tape op surface incl. backward kernels' needs); MLX
   implementation (MlxBuf over mlx_rs::Array, ops stay lazy /
   resident, download = eval + f64 widen) + registration; parity
   unit tests vs DenseArray ops; Linux/wasm builds stay green with
   the Dev arm inert.
3. tape-forward-resident -- NodeData.value becomes TensorHandle;
   eval_tensor_expr forward ops keep intermediates resident under
   device("mlx") via the registry (per-op CPU fallback downloads
   when an op has no device kernel); DELETE
   materialize_tape_on_mlx (the 2x forward). grad_mlx parity
   suites stay green at tolerance.
4. tape-backward-resident -- backward formulas run resident for
   the core op set (matmul, elementwise, softmax/CE, reductions,
   reshape/transpose); structural ops (patchify/take/stack/rotate)
   may download-fallback; grads stay resident end-to-end.
5. optimizer-resident -- adam / momentum_sgd update on handles;
   OptimizerState.buffers hold TensorHandle; params bound under
   device("mlx") upload ONCE and stay resident across train N;
   explicit sync surface: device-block exit, into_array reads,
   and per-step scalar loss download only. GpuEnv/GpuAdamStep
   seam updated or bypassed for MLX.
6. parity-and-bench-gate -- tiny-LM BASE-model train on MLX:
   per-step parity vs CPU at FP32_TOL, 30-step loss-trajectory
   tolerance test, engram-in-chain train parity ride-along (E5
   pre-flight); rerun mlx_vs_cpu; ACCEPTANCE: tiny_lm_train_step
   MLX/CPU ratio > 1.0 (report the lora/thicket ratios too);
   docs/benchmarks.md before/after table.
7. cleanup-and-surface -- retire or demote the bespoke
   demo_forward fast paths if the general path is within reach of
   them (keep documented if still faster); fallback warnings
   still truthful; docs (architecture, future-saga-gpu-training
   marked landed, saga.md entry); demo/pages refresh only if a
   demo's wording or device gating changes. --done.
