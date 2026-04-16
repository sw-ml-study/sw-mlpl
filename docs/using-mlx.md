# Using MLPL with MLX (Apple Silicon)

> **Status:** planned -- Saga 14 (`v0.11` target). Not yet
> shipped. This doc sketches the intended API so users and
> contributors can plan around it; actual code lands with the
> saga. Treat everything below as design, not reference.

## Why MLX

MLX is Apple's array framework for Apple Silicon: lazy graph
execution, automatic operator fusion, unified CPU+GPU memory, and
no kernel-per-device dance. It's the right first accelerator
backend for MLPL because:

- Apple Silicon dev laptops are the baseline for this project.
- MLX's array type already matches MLPL's `DenseArray` semantics
  (row-major, shape-labeled, broadcast rules agree).
- Lazy execution composes cleanly with MLPL's existing
  tape-based autograd -- the autograd tape builds the graph the
  fuser already wants.

CUDA (Saga 17) is the second accelerator backend. See
`docs/using-cuda.md`. The plan is for both backends to share the
same MLPL surface -- a program that trains on MLX should train on
CUDA without source changes.

## Saga 14 shape

`docs/plan.md` breaks Saga 14 into roughly three phases:

1. **`mlpl-mlx` runtime target.** Sibling to `mlpl-rt`. Re-exports
   the MLX-backed equivalents of every `mlpl-rt` primitive --
   `mlx_rt::matmul`, `mlx_rt::reduce_add`, etc. A feature flag on
   the `mlpl` facade crate selects the backend.
2. **Device placement + movement.** A `device("mlx")` /
   `device("cpu")` scoped form in the language, mirroring PyTorch's
   `.to(device)` but reifying the choice in the AST so the
   compiler can fuse across it. Labels and shapes propagate across
   the device boundary unchanged.
3. **Training on MLX.** The Model DSL's `apply`, `grad`, and
   optimizer surface pick up MLX tensors under the `mlx`
   feature. The Saga 13 Tiny LM demo gets an MLX variant that
   trains roughly 10-50x faster on an M-class laptop than the
   CPU interpreter or compiled-to-rust path.

## Intended API (not shipped)

```mlpl
# Place a fresh tensor on the MLX device
X : [batch, feat] = device("mlx") { randn(7, [1024, 64]) }

# Existing Model DSL, transparently dispatched to MLX
mdl = chain(linear(64, 256, 0), relu_layer(), linear(256, 64, 1))
Y   = apply(mdl, X)

# Training loop, identical surface to the CPU version
train 100 {
  adam(cross_entropy(apply(mdl, X), targets), mdl,
       0.01, 0.9, 0.999, 0.00000001);
  loss_metric = cross_entropy(apply(mdl, X), targets)
}
```

The intent is that swapping `device("mlx")` for `device("cpu")`
(or dropping it) is the only change needed between MLX-accelerated
and CPU-interpreted runs. Shape checking, labels, and the
`:describe` / `:vars` output stay identical.

## Fine-tuning use case

> **Status:** even more speculative than the above. Landing this
> depends on Saga 15 (LoRA / QLoRA / quantization).

Fine-tuning a small pretrained LM with LoRA on Apple Silicon is
the planned flagship MLX demo. Rough shape of the workflow:

1. Load a pretrained checkpoint into MLPL's Model DSL (requires a
   checkpoint format -- Saga 13 deliberately deferred this).
2. Freeze the base-model parameters via a flag on `param[...]`
   (deferred to Saga 15).
3. Attach low-rank adapters on the attention layers (`lora(...)`
   as a new layer kind in the Model DSL, also Saga 15).
4. Train the adapters with `adam` on MLX.

Until Sagas 14 + 15 land, the practical fine-tuning path is the
CPU Saga 13 Tiny LM, which trains in seconds on a tiny corpus and
is a good place to learn the API shape.

## What you can do today

- **Prototype the shape of a model** with the current Model DSL.
  Use `:describe mdl` and `:vars` to confirm shapes and label
  propagation -- exactly what will transfer to MLX once that
  backend lands.
- **Measure the CPU baseline.** `cargo bench -p mlpl-bench` runs
  the interpreter vs the compile-to-Rust path. The MLX numbers
  will land in the same harness when Saga 14 ships, so you'll
  have a direct comparison.
- **Read `docs/saga.md`** for the current saga status. If Saga 14
  has flipped to `IN PROGRESS` the API in this doc may already be
  stale -- the saga's per-step prompts are the authoritative
  source until the retrospective milestone doc replaces them.

## Why *not* just use MLX directly from Rust?

You can. `mlx-rs` exists. MLPL's value proposition is:

- **One source, many backends.** The same `.mlpl` source that
  trains on CPU trains on MLX trains on CUDA (Saga 17) trains in
  WASM (compile-to-rust, today). No duplicate codebases.
- **Labeled shapes and structured errors.** Writing MLX directly
  means carrying positional shapes in your head again.
- **The REPL surface.** `:describe model` and the tutorial lesson
  flow don't exist on top of raw MLX.

When Saga 14 ships, `docs/compiler-guide.md` will grow a section
on the `mlpl-mlx` feature flag. Until then: CPU-first, and watch
`docs/plan.md`.

## Related

- `docs/plan.md` -- Saga 14 plan (MLX backend)
- `docs/saga.md` -- live saga status
- `docs/using-cuda.md` -- the other accelerator backend (Saga 17)
- `docs/benchmarks.md` -- where MLX numbers will land once the
  backend ships
