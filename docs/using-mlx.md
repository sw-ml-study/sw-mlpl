# Using MLPL with MLX (Apple Silicon)

> **Status:** reference. Shipped in Saga 14 (v0.11.0). For the
> design-sketch history that predated the shipped saga, see the
> "Retrospective" section at the bottom of this doc.

## Why MLX

MLX is Apple's array framework for Apple Silicon: lazy graph
execution, automatic operator fusion, unified CPU+GPU memory, and
no kernel-per-device dance. It's MLPL's first accelerator backend
because:

- Apple Silicon dev laptops are the baseline for this project.
- MLX's array type matches MLPL's `DenseArray` semantics
  (row-major, shape-labeled, broadcast rules agree).
- Lazy execution composes cleanly with MLPL's tape-based
  autograd -- the autograd tape names the same computation the
  MLX fuser wants to see.

CUDA (Saga 17) is the planned second accelerator backend. See
`docs/using-cuda.md`. Both backends share the same MLPL surface:
a program that trains on MLX trains on CUDA without source
changes.

## What shipped (Saga 14)

Three phases, delivered as ten steps:

1. **`mlpl-mlx` runtime target.** Sibling to `mlpl-rt` under
   `crates/mlpl-mlx`. Re-exports MLX-backed equivalents of every
   forward primitive MLPL exercises -- `matmul`, `add`/`sub`/
   `mul`/`div`/`neg`, `exp`/`log`/`tanh`/`sigmoid`/`relu`,
   `reshape`/`transpose`, reductions (`reduce_mul`/`mean`/
   `argmax` axis-aware), `softmax`/`log_softmax`, and a fused
   `cross_entropy`. Every primitive parity-tested vs the CPU
   path within fp32 tolerance. `mlx-rs` is an optional
   dependency behind a `mlx` Cargo feature.
2. **Device placement + movement.** A `device("mlx") { body }`
   scoped form in the language, mirroring `experiment`'s shape.
   A `to_device(x, "mlx")` / `to_device(x, "cpu")` builtin
   moves a tensor's device tag (and walks the param tree when
   applied to a model). Cross-device ops raise the typed
   `EvalError::DeviceMismatch { op, expected, actual }` rather
   than panicking.
3. **Training on MLX.** `apply(model, X)`, `grad(expr, wrt)`,
   `adam`, `momentum_sgd`, `train N { body }`, and
   `experiment "name" { }` all work unchanged inside
   `device("mlx") { }`. The autograd tape's forward values
   re-materialize through `mlpl-mlx` before CPU backward
   formulas compute gradients; gradients and optimizer updates
   match the CPU path within fp32 tolerance for the full Saga 11
   model zoo (linear, chain, residual, rms_norm, attention,
   causal_attention, embed, activations) and a Tiny-LM
   composition end-to-end.

## Shipped API

```mlpl
# A fresh tensor on MLX. Inside the device block, randn dispatches
# through mlpl-mlx; outside, behaviour is unchanged.
device("mlx") { randn(7, [1024, 64]) }

# Build a model whose params live on MLX. The block tags every
# allocated param with the "mlx" device at creation time, so no
# per-param to_device call is needed.
device("mlx") {
  m = chain(linear(64, 256, 0), relu_layer(), linear(256, 64, 1))
}

# Move a tensor to the device before calling apply.
x = randn(0, [8, 64])
to_device(x, "mlx")

# apply(model, X) routes matmul/softmax/add through mlpl-mlx
# when the active scope is MLX and the model's params agree.
device("mlx") { apply(m, x) }

# Training loop, identical surface to the CPU version.
device("mlx") {
  train 100 {
    adam(cross_entropy(apply(m, X), Y), m,
         0.01, 0.9, 0.999, 0.00000001);
    loss_metric = cross_entropy(apply(m, X), Y)
  }
}
```

Running the above:

```bash
# On Apple Silicon, enable the mlx feature:
cargo run -p mlpl-repl --features mlx -- -f demos/tiny_lm_mlx.mlpl

# On any other host (or without the feature), the device("mlx")
# block emits a one-time "falling back to CPU" warning and the
# rest of the program runs unchanged. Correctness is preserved;
# speed is the CPU baseline.
```

## Performance

Saga 14 step 008 measured MLX vs CPU wall clock on Apple Silicon
using a Criterion harness (`crates/mlpl-bench/benches/
mlx_vs_cpu.rs`). The measured numbers:

| Workload | CPU warm | MLX warm | Warm ratio |
|---|---:|---:|---:|
| `reshape_reduce_100x100` | 68.5 us | 81.1 us | 0.84x |
| `tiny_lm_train_step` (V=60, d=16, T=8) | 619 us | 2.36 ms | 0.26x |

**MLX is currently slower than CPU at Tiny-LM scale.** The 5x
warm-path speedup gate the saga plan set was missed. Four
compounding bottlenecks:

1. **f32/f64 round-trip on every op.** Every MLX call copies
   input data f64 -> f32 into a fresh MLX array and copies the
   result back. At 10 k elements or smaller the copy is a
   measurable fraction of total work.
2. **No graph fusion.** Each primitive wraps its call in
   `.eval()`, so MLX's lazy-graph advantage is never given more
   than one node at a time.
3. **Tape re-materialization.** `grad(expr, wrt)` runs the CPU
   forward to build the autograd tape, then walks the tape a
   second time to re-materialize forward values on MLX. That's
   2x the forward work per gradient.
4. **Small inner dimensions.** Tiny LM at d=16/32 is dominated
   by kernel-launch overhead. Same architecture at d=256+ would
   flip the ratio.

Correctness is unaffected -- every MLX-gated parity test passes
(forward, reductions, softmax, cross_entropy, autograd
gradcheck, Adam/momentum_sgd, train + last_losses, Tiny LM
forward + training loop). A dedicated "MLX throughput" step is
the natural follow-up; see `docs/benchmarks.md` for the
bottleneck breakdown and the optimization path.

## Fine-tuning use case

> **Status:** planned, not shipped. Depends on Saga 15 (LoRA /
> QLoRA / quantization) and a checkpoint format.

Fine-tuning a small pretrained LM with LoRA on Apple Silicon is
the planned flagship MLX demo. Rough shape of the workflow when
Saga 15 lands:

1. Load a pretrained checkpoint into MLPL's Model DSL (requires
   a checkpoint format -- Saga 13 deliberately deferred this).
2. Freeze the base-model parameters via a flag on `param[...]`
   (Saga 15).
3. Attach low-rank adapters on the attention layers (`lora(...)`
   as a new layer kind, Saga 15).
4. Train the adapters with `adam` inside `device("mlx") { }`
   (uses Saga 14's surface; works today on small-from-scratch
   models, will work on loaded checkpoints once Saga 15 ships).

Until Saga 15 ships, the practical on-MLX path is the Saga 13
Tiny LM, which trains in seconds on a tiny corpus and exercises
the full MLX surface end-to-end.

## What you can do now

- **Run the Tiny LM on MLX.** `demos/tiny_lm_mlx.mlpl` is the
  Saga 13 body wrapped in `device("mlx") { }`, identical seeds,
  dataset, and hyperparams. Loss curve shape matches the CPU
  variant within fp32 tolerance.
- **Prototype a model under `device("mlx") { }` in the REPL.**
  Use `:describe mdl` and `:vars` to confirm shapes and labels
  propagate across the boundary. Cross-device mistakes surface
  as `EvalError::DeviceMismatch` with a clear message.
- **Measure your own workload.** `cargo bench -p mlpl-bench
  --features mlx` on an M-class laptop reports CPU vs MLX
  warm-path medians plus a one-shot cold timing.
- **Read `docs/benchmarks.md`** for the step-008 measurement
  and the bottleneck analysis; it is the authoritative source
  on where MLX is and isn't a win today.

## Why *not* just use MLX directly from Rust?

You can. `mlx-rs` exists. MLPL's value proposition is:

- **One source, many backends.** The same `.mlpl` source that
  trains on CPU trains on MLX (today, Saga 14) and will train
  on CUDA (Saga 17). No duplicate codebases.
- **Labeled shapes and structured errors.** Writing `mlx-rs`
  directly means carrying positional shapes in your head again.
  MLPL's named axes and `EvalError::ShapeMismatch` stay on the
  MLX path.
- **The REPL surface.** `:describe model`, `:experiments`,
  `attention_weights(model, X)`, the heatmap/scatter viz
  helpers, and the tutorial lesson flow don't exist on top of
  raw MLX.

## Retrospective (what shipped vs the sketch)

The Saga 14 sketch (the contents of this doc before the
retrospective) predicted three phases and a 10-50x speedup on
an M-class laptop. What actually landed:

- **API surface.** The sketch's `device("mlx") { body }` form,
  the `to_device` movement helper, and the "same source, drop
  the wrapper for CPU" invariant all shipped as described. The
  scoped form mirrors `experiment "name" { body }`'s shape
  down to nesting rules (`experiment { device { ... } }` and
  `device { experiment { ... } }` both compose correctly).
- **Cross-device safety.** Not in the sketch. Shipped as the
  typed `EvalError::DeviceMismatch` variant, raised by
  `apply(model, X)` when the model's params and the input
  disagree on device tag.
- **Autograd on MLX.** Sketch said "lazy execution composes
  cleanly with MLPL's tape-based autograd." Reality was
  messier: MLX's own `vjp`/`grad` helpers expect an
  `Fn(Array) -> Array` closure incompatible with MLPL's
  `Tape`/`NodeId` graph. Saga 14 step 006 chose option (b) --
  keep the CPU tape structure, re-materialize forward values
  on MLX after the tape is built, let CPU backward formulas
  operate on MLX-rounded values -- and documented the choice.
  Gradients match CPU within 1e-4 across every tape primitive.
- **Optimizer state.** Sketch said `OptimizerState` would hold
  "MLX-typed tensors when the surrounding scope is MLX."
  Shipped: `OptimizerState` stays as CPU-resident `DenseArray`
  values holding MLX-rounded numbers. The distinction is a
  perf optimization, not correctness, and was deferred as a
  follow-up.
- **Tiny LM speedup.** Sketch said 10-50x. Reality: 0.26x
  (MLX slower) on a Tiny-LM-sized training step at the current
  dispatch granularity. The bottlenecks are understood and
  documented; a targeted "MLX throughput" step is the natural
  follow-up. See the Performance section above.
- **What the sketch got right.** Source-level parity is real:
  the same `.mlpl` program runs on either device with the same
  shapes, labels, loss curves, and parameter updates within
  fp32 tolerance. The platform thesis of "one source, many
  backends" holds up end-to-end.

## Related

- `docs/plan.md` -- Saga 14 plan + dependency graph
- `docs/saga.md` -- saga-by-saga ship log
- `docs/using-cuda.md` -- the other accelerator backend (Saga 17)
- `docs/benchmarks.md` -- "Saga 14: MLX vs interpreter CPU"
  section with measured numbers and the perf follow-up plan
- `demos/tiny_lm_mlx.mlpl` -- the Saga 13 Tiny LM wrapped in
  `device("mlx") { }`
- `crates/mlpl-mlx/` -- the runtime target crate
- `crates/mlpl-eval/src/device.rs` -- dispatch helper + device
  stack + autograd tape rematerialization
