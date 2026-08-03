# Benchmarking MLPL

MLPL has two hot paths -- the tree-walking interpreter in
`mlpl-eval` and the Rust code `mlpl-lower-rs` emits into
`mlpl-rt` at Rust compile time. When the language design goal is
"does the compile path earn its keep", the only answer that
carries weight is a reproducible benchmark. This doc explains
where those benchmarks live, how to run them, what the current
numbers are on the author's laptop, and what the numbers do and
don't say.

## TL;DR

```bash
cargo bench -p mlpl-bench
```

That's the whole story. Six workloads, each run through both the
interpreter and the compiled path, medians reported by Criterion.
HTML reports land in `target/criterion/`.

## Where the harness lives

- `crates/mlpl-bench/` -- new in v0.10. Dev-only crate; not
  linked into any binary.
- `crates/mlpl-bench/src/lib.rs` -- declares a
  `WORKLOADS: &[(&str, &str)]` constant. Each entry is
  `(name, mlpl_source)` where `mlpl_source` must stay inside the
  lowered subset (see `docs/compiling-mlpl.md` "Out of scope").
- `crates/mlpl-bench/build.rs` -- at `cargo bench` build time,
  lowers every workload through `mlpl_parser::lex + parse +
  mlpl_lower_rs::lower` and emits one
  `pub fn case_<name>() -> DenseArray` per workload to
  `$OUT_DIR/compiled_cases.rs`. The bench thus measures real
  lowered code, not a hand-written stand-in.
- `crates/mlpl-bench/benches/interp_vs_compiled.rs` -- Criterion
  harness. For each workload it pre-parses the source once
  (parsing happens at Rust compile time for the compile path;
  double-counting it would make interpreter numbers look worse
  than they are), then benches two functions in a named group:
  `<name>/interp` and `<name>/compiled`.

## What the workloads exercise

| Workload | Source | What it stresses |
|---|---|---|
| `scalar_tight` | `1 + 2 * 3 - 4` | Parser dispatch + tree-walk overhead on pure scalar ops. The interpreter's fastest path; compiled's biggest relative win is vs dispatch. |
| `small_array_arith` | `reduce_add([1,2,3,4,5] * 10 + [0,1,2,3,4])` | Short broadcast + reduction. Allocator + per-op dispatch, not memory traffic. |
| `iota_reduce_100` | `reduce_add(iota(100))` | One medium sweep. Memory-touching but linear. |
| `reshape_reduce_100x100` | `m = reshape(iota(10000), [100, 100]); rows = reduce_add(m, 0); cols = reduce_add(m, 1); reduce_add(rows) + reduce_add(cols)` | The `docs/milestone-compile-to-rust.md` baseline workload. Dominated by memory traffic -- both paths hit the same `reduce` in `mlpl-rt`. |
| `matmul_16x16` | `a = reshape(iota(256), [16, 16]); b = reshape(iota(256) + 1, [16, 16]); reduce_add(matmul(a, b))` | Matmul-heavy; inner kernel identical on both paths. |
| `transpose_chain_10x10` | `m = reshape(iota(100), [10, 10]); reduce_add(transpose(m) + m)` | Transpose + elementwise + reduce chain. Multiple small ops where per-op overhead matters. |

Every workload stays inside the lowered subset on purpose --
`repeat`/`train`/`grad`/the Model DSL are interpreter-only today
and would fail `build.rs` with `LowerError::Unsupported`.

## Reference numbers

Measured on the author's M-class laptop (`cargo bench -p
mlpl-bench -- --measurement-time 3`). Do not treat these as
portable; reproduce locally for your machine.

| Workload | Interpreter | Compiled | Speedup |
|---|---:|---:|---:|
| `scalar_tight` | 273 ns | 124 ns | 2.2x |
| `small_array_arith` | 1.29 us | 402 ns | 3.2x |
| `iota_reduce_100` | 994 ns | 150 ns | 6.6x |
| `reshape_reduce_100x100` | 93.7 us | 57.8 us | 1.6x |
| `matmul_16x16` | 6.88 us | 3.53 us | 1.9x |
| `transpose_chain_10x10` | 3.48 us | 836 ns | 4.2x |

**What the range means.** The compile path wins most on workloads
where the interpreter's per-op dispatch is a measurable fraction
of total time (`scalar_tight`, `small_array_arith`,
`iota_reduce_100`, `transpose_chain_10x10`). On workloads
dominated by memory traffic inside identical inner kernels
(`reshape_reduce_100x100`, `matmul_16x16`), the floor is shared
and the ratio shrinks -- compiled still wins, but the interpreter
isn't paying much overhead per useful byte moved.

**Historical note.** `docs/milestone-compile-to-rust.md` originally
quoted **9.05x** on the `reshape_reduce_100x100` workload
(interpreter 479us -> compiled 53us). The compiled number has
barely moved (53us -> 58us on my laptop); what shrank the ratio
is the interpreter: 479us -> 94us, roughly 5x faster after
subsequent interpreter-side work landed. The headline in the
milestone doc is a stale snapshot, not a regression.

## How to run

```bash
# Full run (~45s with default Criterion budget)
cargo bench -p mlpl-bench

# Shorter, less statistically rigorous
cargo bench -p mlpl-bench --bench interp_vs_compiled -- \
    --warm-up-time 1 --measurement-time 3

# One workload only
cargo bench -p mlpl-bench --bench interp_vs_compiled -- matmul_16x16

# Build-only sanity check (useful in CI; does not run)
cargo bench -p mlpl-bench --no-run
```

Criterion writes HTML reports with distribution plots to
`target/criterion/<workload>/report/index.html`. Open one to see
the sample density, confidence interval, and regression check vs
a previous run.

## Adding a workload

1. Append `(name, source)` to `WORKLOADS` in
   `crates/mlpl-bench/src/lib.rs`.
2. Mirror the same pair in `crates/mlpl-bench/build.rs` (the
   constant is duplicated; a build script can't depend on its own
   crate's lib).
3. Run `cargo bench -p mlpl-bench --no-run`. If `build.rs` panics
   with `LowerError::Unsupported`, the source uses something the
   compile path doesn't lower yet; pick a different workload or
   narrow it.
4. Names must be snake_case and unique (they become Rust
   identifiers and Criterion group names).

## What is *not* here

- **No CI wiring.** Shared CI runners are too noisy for
  meaningful benchmark numbers. Run locally.
- **No regression gate.** Criterion supports `--save-baseline`
  and `--baseline <name>`; a future step can turn that into a
  `benchcmp`-style PR check when someone cares enough.
- **No interpreter-only workloads.** `repeat`/`train`/`grad`/the
  Model DSL aren't lowered, so the "compiled" column has no
  entry. Once a later saga extends lowering, those workloads can
  land here.

## Saga 14: MLX vs interpreter CPU

A second Criterion harness lives at
`crates/mlpl-bench/benches/mlx_vs_cpu.rs` and runs the same
interpreter code path twice per workload: once on the CPU
runtime and once wrapped in `device("mlx") { ... }` so ops
dispatch through `mlpl-mlx`. Triple-gated on macOS + aarch64 +
the `mlx` Cargo feature; `cargo bench -p mlpl-bench` on any
non-MLX host skips this binary entirely.

```bash
# Full MLX harness (roughly 40s with Criterion's default budget)
cargo bench -p mlpl-bench --features mlx --bench mlx_vs_cpu
```

### Saga E4 step 001 baseline: Metal ON (Apple Silicon, 2026-08-01)

E4 (mlx-persistent-tensors) turned the vendored mlx-rs `metal`
feature ON, so MLX now executes on the GPU instead of Apple's
Accelerate CPU BLAS. Same benchmarks, same per-op dispatch path
as the 2026-04-21 table below -- ONLY the backend changed. The
result is the saga's motivation in one table: with per-op
upload + `.eval()` + download, a real GPU is 10-30x WORSE than
Accelerate at demo scale, because kernel-launch latency (tens of
microseconds per op, paid dozens of times per step) dwarfs the
arithmetic at d=8-16.

| Workload | CPU warm | MLX warm (Metal) | Ratio | Prior (Accelerate) |
|---|---:|---:|---:|---:|
| `reshape_reduce_100x100` | 69.3 us | 315.8 us | **0.22x** | 0.84x |
| `tiny_lm_train_step` | 939.9 us | 116.6 ms | **0.008x** | 0.26x |
| `neural_thicket_variant_loop` | 1.159 ms | 79.25 ms | **0.015x** | 0.25x |
| `lora_finetune_step` | 235.9 us | 16.06 ms | **0.015x** | 0.15x |

Cold (first-call, includes Metal shader/pipeline setup):
reshape 129.9 us / 32.2 ms, tiny_lm 1.34 ms / 57.9 ms, thicket
1.41 ms / 127.1 ms, lora 355 us / 19.2 ms.

Reading: the four costs documented below did not change -- Metal
just reprices cost (2) (no graph fusion) and cost (1) (per-op
round-trips) from "function call + memcpy" to "GPU submission +
PCIe-class latency". Fixing them (persistent device tensors, one
graph per training step, resident optimizer state) is exactly
saga E4 steps 2-5; this table is the honest "before".

Also landed with the flip: a process-wide MLX submission lock
(`mlpl_mlx_rt::mlx_op_lock`) -- the Metal backend SIGSEGVs under
concurrent submissions from parallel test threads (Accelerate,
being CPU code, never did). Every mlpl-mlx-rt op and both
gpu_step entries hold it; sibling-crate unit tests serialize on
per-crate `MLX_TEST_LOCK`s (the mlpl-mlx-train idiom).

### Saga E4 step 008: resident-tape AFTER (Apple Silicon, 2026-08-01)

E4 steps 2-7 landed the TensorHandle seam, the resident
forward/backward tape, the resident optimizer (one tape per step,
weights + Adam moments stay on the device), and step 008 added
seam instrumentation plus two backward fixes it exposed
(transpose/reshape gradients and scalar-broadcast binary
gradients now stay lazy on the device). Warm numbers, same
harness:

| Workload | CPU warm | MLX warm (Metal) | Ratio | Step-001 ratio |
|---|---:|---:|---:|---:|
| `reshape_reduce_100x100` | 69.6 us | 316.6 us | 0.22x | 0.22x |
| `tiny_lm_train_step` (d=16) | 185.2 us | 4.76 ms | 0.039x | 0.008x |
| `tiny_lm_train_loop30` (warm, per step) | 143 us | 4.75 ms | 0.030x | n/a |
| `tiny_lm_train_step_d256` | 40.3 ms | 13.6 ms | **2.96x** | n/a |
| `neural_thicket_variant_loop` | 1.11 ms | 78.6 ms | 0.014x | 0.015x |
| `lora_finetune_step` | 140 us | 2.94 ms | 0.048x | 0.015x |

Two changes moved BOTH columns: MLX went 116.6 ms -> 4.76 ms per
tiny-LM step (24x, residency), and the CPU went 940 us -> 185 us
(5x -- the step-006 batched-gradient fix removed the per-param
tape rebuild for everyone).

The seam counters (printed by the bench, `seam/` lines) explain
the residual gap at tiny scale. One warm d=16 train step is now:

    uploads=16 downloads=18 submits=226 cpu_fallbacks=1

The single remaining fallback is cross-entropy backward (the
documented CPU kernel). Nothing re-uploads the model; the floor
is per-op DISPATCH: ~226 lazy op submissions through the mlx-rs
FFI (each holding the process-wide Metal lock) plus ~18 graph
forces per step. At d=16 the kernels are trivial, so that floor
is the whole cost -- and no residency work can remove it.

Consequently the ratio is a function of model size (50-step warm
`train` loops via the repl, V=280):

| Scale | CPU | MLX | Winner |
|---|---:|---:|---|
| d=32, T=32 (live-demo size) | 0.11 s | 0.29 s | CPU 2.6x |
| d=128, T=64 | 0.39 s | 0.30 s | MLX 1.3x |
| d=256, T=128 | 1.72 s | 0.34 s | MLX 5x |

MLX wall-clock is nearly FLAT across those scales -- pure
dispatch bound -- while the CPU grows with the arithmetic.

**Acceptance verdict:** the E4 gate (tiny-LM train on MLX faster
than interpreted CPU) is MET from roughly d>=128 upward and is
pinned in-repo by `tiny_lm_train_step_d256` (2.96x). At the
original d=16 bench slice the ratio is 0.039x and CANNOT cross
1.0 without reducing submissions per step (op fusion / mlx
compile / a batched step graph) -- that is generation-speed-track
scope (docs/future-sagas-queue.md Track 2), not residency scope.
Of the four costs documented below: (1) per-op round-trips are
GONE (uploads/downloads above), (3) tape re-materialization is
GONE (one resident tape per step), (4) optimizer re-upload is
GONE (resident moments + witness cache); (2) no-graph-fusion
remains, repriced as the 226-submission dispatch floor.

### Saga E5: engram-in-chain on the resident seam (Apple Silicon, 2026-08-03)

E5 put the Engram layer on the E4 TensorHandle seam. The
selection-matmul gather runs resident BOTH ways (`sel @ memory`
gathers; `sel^T @ upstream` is an exact scatter-ADD -- a device
gather was evaluated and rejected because the vendored mlx-rs has
no scatter-add kernel), concat joined the device op set
(`DeviceShapeOps`), and the parity ledger is pinned by tests:
bit-exact hashing under device blocks, 10-step trajectory drift
0.000000, duplicate-address accumulation exact.

Warm engram-in-chain step profile (per step, `seam/` lines):

| Milestone | uploads | downloads | submits | cpu_fallbacks |
|---|---:|---:|---:|---:|
| E5 baseline | 23 | 30 | 330 | 3 (concat fwd/bwd + CE) |
| + dev-concat | 19 | 27 | 332 | 1 (CE backward only) |
| + loss reporting | 19 | 28 | 332 | 1 |

(The +1 download is the per-step loss scalar -- the one reporting
sync the E4 contract budgets; it also fixed four demos whose loss
curves were silently flat.)

Bench numbers (fresh env per iteration, cold model build + upload
included in the MLX column):

| Workload | CPU warm | MLX warm | Ratio |
|---|---:|---:|---:|
| `engram_train_step` (d=8, 128 slots) | 885 us | 8.58 ms | 0.10x |
| `engram_train_step_d64` (512 slots) | 9.08 ms | 9.41 ms | 0.96x |
| repl 50-step loop, d=128 (1024 slots) | 0.59 s | 0.44 s | **~1.4x** |

Same story as the base tiny-LM: the seam profile is FLAT across
scales (19/28/332/1 at d=8 and d=64 alike -- pure dispatch
floor), so MLX loses at toy widths and crosses over near d=128.
The engram layer adds no extra seam crossings beyond the one CE
fallback every model pays.

### Measured numbers (Apple Silicon, 2026-04-21)

Cold timings are one-shot wall-clock prints from the harness and
include MLX's first-call compile overhead. Warm timings are
Criterion's steady-state medians after a 3s warm-up.

| Workload | CPU cold | MLX cold | CPU warm | MLX warm | Warm ratio |
|---|---:|---:|---:|---:|---:|
| `reshape_reduce_100x100` | 206 us | 347 us | 68.5 us | 81.1 us | **0.84x** (MLX slower) |
| `tiny_lm_train_step` | 769 us | 2.60 ms | 619 us | 2.36 ms | **0.26x** (MLX slower) |
| `neural_thicket_variant_loop` | 838 us | 3.12 ms | 767 us | 3.01 ms | **0.25x** (MLX slower) |
| `lora_finetune_step` | 208 us | 1.45 ms | 164 us | 1.11 ms | **0.15x** (MLX slower) |

`tiny_lm_train_step` is one Adam step (forward + cross_entropy +
backward + Adam update) on a Saga 13 Tiny LM-shaped slice scaled
to V=60, d=16, T=8, single-head causal attention. The full
`demos/tiny_lm_mlx.mlpl` is V=280, d=32, T=32, 200 steps --
roughly 20x more work per iteration, so its warm-path
performance trends in the same direction but amortizes more of
the per-op overhead.

`neural_thicket_variant_loop` (added Saga 20 step 005) is 16
perturbation variants scored through a Tiny LM-shaped base
(V=32, d=8, T=16, single-head causal attention) inside one
`device("mlx") { ... }` block: for each variant, a
`clone_model` + `perturb_params` + `apply` + `cross_entropy` +
`scatter` cycle. Each iteration does 16 forwards (and no
training), so the ratio is driven by inference throughput
rather than by the tape-rematerialization cost that dominated
`tiny_lm_train_step`. Yet the MLX-vs-CPU ratio is essentially
unchanged (0.25x vs 0.26x), which is consistent with the
bottleneck analysis below: at this inner dimension, per-op
kernel launch + f32 round-trip overhead swamps the forward
arithmetic regardless of whether a backward/Adam pass also
runs.

`lora_finetune_step` (added Saga 15 step 005) is one Adam
step through a LoRA-wrapped Tiny LM (V=16, d=8, T=4,
rank=2) with the base auto-frozen inside `lora()`. The MLX
ratio (0.15x) is noticeably worse than the non-LoRA
`tiny_lm_train_step` (0.26x) or the non-training
`neural_thicket_variant_loop` (0.25x). The direct cause:
each `LinearLora` forward is `X @ W + (alpha/rank) * X @ A
@ B + b`, which launches three matmuls + one scalar
elementwise multiply per wrapped linear instead of one
matmul per linear. At d=8 the extra kernel launches and
f32 round-trips swamp the arithmetic savings from the
low-rank factorization -- the adapter matmuls are tiny
(shape `[T, rank]` and `[rank, out]`, rank=2) and give MLX
no room to amortize. The same forward on a d=512 model
would flip the ratio: both `X @ W` and `X @ A @ B` would
be matmul-FLOP-bound and MLX's Metal kernels would
dominate. Bottleneck categories unchanged from the Saga 14
breakdown below; LoRA just compounds them because it adds
ops per layer.

### Go/no-go gate result: MISS

The Saga 14 plan set **5x warm-path speedup** as the go/no-go
gate for step 008, with a 10-50x target per `docs/using-mlx.md`.
Measured MLX performance is **below parity** on both workloads
at Tiny LM scale: about 0.84x on the reshape+reduce and 0.26x on
the training step. The step-008 prompt allows this outcome --
the plan explicitly says "do not block the saga on hitting 10x;
ship the MLX demo anyway with the honest number documented, and
open a follow-up step for optimization."

### Why MLX is currently slower at this scale

Four compounding costs, all diagnosable from the current
`mlpl-mlx` dispatch path:

1. **f32 <-> f64 round-trip on every op.** `common::dense_to_mlx`
   casts input data f64 -> f32 and allocates a fresh MLX array;
   `mlx_to_dense_data` does the reverse on the way out. At
   100x100 (10 k elements) or Tiny LM-slice sizes (~1 k
   elements per op), the copy is a noticeable fraction of total
   work; for workloads that would dominate the matmul FLOPs on
   a GPU (think 1024x1024), the copy becomes negligible.

2. **No graph fusion.** Each primitive -- `matmul`, `softmax`,
   `add`, etc. -- wraps the MLX call in an `eval()` that
   materializes immediately. MLX's lazy graph (its main
   performance advantage over eager frameworks) is never given
   more than one node at a time. A proper backend would submit
   a sequence of ops and evaluate once per training step;
   designing that interface is a follow-up.

3. **Saga 14 step 006 tape re-materialization.** Inside
   `grad(expr, wrt)` we compute the forward on CPU to build the
   autograd tape, *then* walk the tape and recompute every node
   on MLX to give backward MLX-rounded values. That's 2x the
   forward work. Option (a) -- leaning on
   `mlx_rs::transforms::grad` -- would cut that in half but
   requires rewriting the tape structure, which the step-006
   commit explicitly deferred as a future optimization.

4. **Small inner dimensions on Tiny LM.** d=16 (or d=32 in the
   full demo) gives a [4, 16] @ [16, 16] matmul where the MLX
   kernel launch overhead is comparable to the arithmetic. The
   same architecture at d=256 or d=512 (a real small LLM) would
   flip the ratio decisively; we just are not running anything
   that big in this saga.

### What ships anyway

- `demos/tiny_lm_mlx.mlpl` -- the Saga 13 Tiny LM body wrapped
  in `device("mlx") { ... }`. Loss curve matches the CPU path
  within fp32 tolerance (validated by a micro-variant parity
  test in `crates/mlpl-eval/tests/tiny_lm_mlx_demo_tests.rs`).
  Correctness is proved; speed is not.
- `mlpl-bench` MLX row -- reproducible numbers for future
  optimization work to target.
- Every MLX-gated parity test across the saga (matmul,
  reductions, softmax, cross_entropy, Tiny LM forward, autograd
  gradcheck, optimizer step) continues to pass, so the
  correctness story is complete.

## Saga 29: Vision Transformer multi-head thorough demo

`demos/vit_multihead_thorough.mlpl` is the deepest end-to-end
ViT demo MLPL ships: a four-head attention block trained for
200 adam steps on a balanced 20-image cat-vs-dog subset of
`pets_tiny`, wrapped in `device("mlx") { ... }` so the forward
dispatches through `mlpl-mlx` when the binary is built with
`--features mlx` on Apple Silicon.

### Measured numbers (Apple Silicon, 2026-05-19, release build, CPU fallback)

| Metric | Value |
|---|---:|
| Wall time (no-mlx-feature CPU fallback, release) | ~285 s |
| Training accuracy (200 steps, 20 images, seed=23/17/31/37) | 1.0 |
| Final cross-entropy loss | < 1e-3 (overfit to the 20-image set) |

The MLX-feature build was not exercised in the step-015
session because `--features mlx` requires an Xcode-enabled
toolchain that wasn't available on the dev host at the time.
The same demo file is the regression baseline for the planned
Saga R2 (CUDA-as-a-service) -- a future session on an MLX-built
binary should re-run it and document the device("mlx") timing
delta here.

### Bottleneck composition

The bulk of the wall time is the AST-walking interpreter
running the inlined forward expression 200 times. Per step the
work is:

- `patchify(X, 16)` -> reshape [20, 16, 768]
- 768 -> 128 patch linear
- multi-head `attention(128, 4, ...)` -- 4 heads each running
  scaled-dot-product on [16, 32] slabs, joined via the new
  `Tensor::stack` op (Saga 29 step 013)
- `take` of position 0 for CLS-like pooling
- 2-layer classifier (128 -> 64 -> 2)
- cross_entropy backward + adam param update

The forward + backward traversal walks ~800 tape nodes per
step. The dominant cost is the per-node dispatch through
`mlpl-runtime`, NOT the matmul arithmetic (which would be the
ratio-shifting factor on a GPU).

### Why this demo matters

Step 013 unlocked the multi-head autograd tape; step 014 shipped
the visualization that makes specialization legible; this demo
proves the unlock works end-to-end in the production training
loop, on real images, at a model size large enough to stress
the per-head + per-batch stack paths. The `attention_weights`
output (`[4, 16, 16]`) renders cleanly through
`svg(_, "heatmap_grid")` and shows four distinct learned
attention patterns -- not the uniformly-random ones the
untrained companion demo produces.

### What's deferred to a future step

A dedicated "MLX throughput" step (slotting naturally before
the Saga 14 release) would target the four bottlenecks above.
Most likely first lever: skip the tape re-materialization when
the forward is already MLX-native (cuts one of the two MLX
forward passes per gradient). After that, lifting the f32 round
trips by keeping MLX arrays alive across multiple ops (instead
of materializing on every `eval()`) is the biggest remaining
win. Both are perf optimizations; neither changes numerical
behaviour, so the parity tests will carry forward unchanged.

## Related

- `crates/mlpl-parity-tests/tests/parity_tests.rs` --
  `compiled_speedup_measurement` (gated by `MLPL_PARITY_TESTS=1`)
  is the pre-Criterion one-shot timing used during the
  compile-to-rust saga. It measures one workload and reports the
  ratio to stderr. The `mlpl-bench` harness supersedes it for
  anything beyond a smoke check.
- `docs/compiling-mlpl.md` -- the user-facing compile-path doc,
  including the three-way comparison of interpreter vs `mlpl!`
  macro vs `mlpl build`.
- `docs/milestone-compile-to-rust.md` -- the compile-to-Rust
  saga's retrospective (and the source of the stale 9x headline).
