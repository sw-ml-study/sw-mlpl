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

### Measured numbers (Apple Silicon, 2026-04-21)

Cold timings are one-shot wall-clock prints from the harness and
include MLX's first-call compile overhead. Warm timings are
Criterion's steady-state medians after a 3s warm-up.

| Workload | CPU cold | MLX cold | CPU warm | MLX warm | Warm ratio |
|---|---:|---:|---:|---:|---:|
| `reshape_reduce_100x100` | 206 us | 347 us | 68.5 us | 81.1 us | **0.84x** (MLX slower) |
| `tiny_lm_train_step` | 769 us | 2.60 ms | 619 us | 2.36 ms | **0.26x** (MLX slower) |
| `neural_thicket_variant_loop` | 838 us | 3.12 ms | 767 us | 3.01 ms | **0.25x** (MLX slower) |

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
