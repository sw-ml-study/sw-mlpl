# Changelog

All notable changes to MLPL. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
canonical per-saga retrospectives live in `docs/saga.md` and
`docs/milestone-*.md`.

## v0.12.0 -- Saga 20: Neural Thickets (2026-04-22)

Weight-perturbation research workflow lands in MLPL source.
Four new builtins plus a headline demo compose into a
RandOpt / Neural-Thickets-style specialization search with a
first-class `[family x seed]` heatmap.

### Added

- **`clone_model(m) -> Model`** (step 001). Deep-copies a
  `ModelSpec` tree and allocates a disjoint set of fresh
  param names; device tags (`mlx` vs `cpu`) propagate from
  source to clone. Contract:
  `contracts/eval-contract/clone-model.md`.
- **`perturb_params(m, family, sigma, seed)`** (step 002).
  Adds `sigma * randn(seed + i, shape)` in place to each
  parameter of the named family. Families:
  `all_layers`, `attention_only`, `mlp_only`,
  `embed_and_head`. The "final projection head" is detected
  structurally (last top-level `Linear` child of the
  outermost `Chain`), not by name pattern, so `mlp_only` /
  `embed_and_head` can tell MLP linears apart from the
  vocab projection. Contract:
  `contracts/eval-contract/perturb-params.md`.
- **`argtop_k(values, k)`** (step 003). Index-returning
  companion to the existing `top_k(logits, k)`. Returns the
  k indices of the largest entries in a rank-1 vector,
  sorted by descending value with ties broken by lower
  original index first (stable sort). Contract:
  `contracts/eval-contract/argtop-k.md`.
- **`scatter(buffer, index, value)`** (step 003). Functional
  rank-1 scalar write; returns a copy of `buffer` with the
  single entry at `index` replaced by `value`. Contract:
  `contracts/eval-contract/scatter.md`.
- **`demos/neural_thicket.mlpl`** (step 004, CPU). Full
  end-to-end: train a Saga 13 Tiny LM (V=280, d=32) for 100
  Adam steps, sweep 4 families x 4 seeds = 16 variants,
  score each on a held-out string, heatmap the
  `[family x seed]` specialization, argtop_k the top-K
  specialists, average logits across all 16 for an
  ensemble cross-entropy.
- **`demos/neural_thicket_mlx.mlpl`** (step 005). Same
  source with the variant loop + ensemble under
  `device("mlx") { ... }`. Base training stays on CPU.
- **Criterion bench group `neural_thicket_variant_loop`**
  (step 005) in `crates/mlpl-bench/benches/mlx_vs_cpu.rs`.
- **Web REPL tutorial lesson "Neural Thickets"** (step 006,
  `apps/mlpl-web/src/lessons.rs`). Interactive 4x4 sweep at
  V=8/d=4 so the full heatmap renders in the browser.
- **`docs/using-perturbation.md`** (step 006). User-facing
  retrospective covering the four builtins, the four
  families, the heatmap narrative, honest MLX-vs-CPU
  numbers, parity testing, and the deferred follow-up
  surface.

### Measured

On an M-class laptop (`cargo bench -p mlpl-bench --features mlx
--bench mlx_vs_cpu -- neural_thicket`):

| Path | Cold | Warm |
|---|---:|---:|
| CPU | 838 us | 767 us |
| MLX | 3.12 ms | 3.01 ms |

MLX is **0.25x** of CPU on this workload, essentially
identical to the Saga 14 Tiny-LM training-step ratio
(0.26x). The Neural Thickets workload is inference-only (no
autograd, no tape rematerialization), yet the ratio is
unchanged -- evidence that the bottleneck at Tiny-LM inner
dimensions is per-op kernel launch + f32 round-trip, not
autograd-specific cost. See `docs/benchmarks.md` for the
four-way bottleneck analysis.

Correctness: per-variant losses + ensemble logits agree
with the CPU path elementwise within fp32 tolerance (1e-3),
pinned by `crates/mlpl-eval/tests/neural_thicket_mlx_demo_tests.rs`.

### Refactored

- Extracted `Lesson` struct + `LESSONS` const from
  `apps/mlpl-web/src/tutorial.rs` into a new
  `apps/mlpl-web/src/lessons.rs` module. Keeps both files
  under the sw-checklist 500-LOC budget after adding the
  Neural Thickets lesson.

### Not shipped (deferred follow-ups)

- Depth-aware families (`early_N_layers`, `late_N_layers`).
  Need explicit layer indices in parameter names, which the
  Model DSL does not encode today.
- Low-rank perturbation (`perturb_low_rank`). Sibling
  builtin to `perturb_params`; deferred until a concrete
  use case lands.
- Strict top-K ensembling. CPU demo picks `best_idx` via
  `argtop_k` but averages all 16 variants rather than
  rebuilding only the best four. Rebuilding requires
  selecting the family string per flat index, which needs
  either string-array indexing or an index-to-family lookup.
- Real pretrained checkpoints. The demo runs on a
  from-scratch Tiny LM; Llama-class bases need Saga 15+
  (checkpoint format) or Saga 19 (LLM sidecar).
- Per-iteration counter in `repeat`. `train N { }` binds
  `step`; `repeat N { }` does not. The demo works around
  this with `for i in [0, 1, 2, 3] { ... }` per family.

See `docs/using-perturbation.md` "Not shipped" and
`docs/saga.md` Saga 20 entry.

## v0.11.0 -- Saga 14: MLX backend (2026-04-21)

First accelerator backend for MLPL. A program that trains on
CPU runs on Apple Silicon via MLX without source changes.
Ten steps across five phases; correctness complete,
throughput honest. See `docs/saga.md` Saga 14 entry and
`docs/using-mlx.md` for details.

## Prior releases

See `docs/saga.md` and `docs/status.md` for the full
per-saga history back to v0.1. CHANGELOG entries below this
point are maintained as a forward-looking record from the
v0.12.0 cut onward; earlier per-release notes live in the
release commits (search
`git log --grep "^release("`).
