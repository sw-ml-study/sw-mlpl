# Using MLPL for Weight-Perturbation Research

> **Status:** reference. Shipped in Saga 20 (v0.12.0). For the
> design-sketch history that motivated the saga, see
> `docs/mlpl-for-neural-thickets.md`. This doc is the shipped
> surface + honest retrospective.

## What this is about

Weight perturbation is a simple, non-gradient-based search
move: take a trained model, add Gaussian noise to some subset
of its parameters, and see whether the perturbed variant is
more specialized on a task than the base. The
[Neural Thickets / nt-rs](https://github.com/swcraig/nt-rs)
harness for [RandOpt](https://www.alphaxiv.org/overview/2603.12228.pdfv1)
takes this further: generate N variants across several
perturbation families, score each, pick the top-K, ensemble.

Saga 20 ships the four builtins that let MLPL express this
workflow in ~100 lines of source plus an out-of-the-box
specialization heatmap.

## The four builtins

### `clone_model(m) -> Model`

Deep-copies a `ModelSpec` tree and allocates a fresh, disjoint
set of parameter identifiers. The clone's forward is
bit-identical to the source until someone mutates the clone.
Mutating the clone's params (via `perturb_params`, `adam`,
`env.set`, or anything else) does not touch the source.
Device tags (`mlx` vs `cpu`) propagate from source params to
clone params, so a clone of an MLX-tagged model is itself
MLX-tagged and `apply(clone, X)` inside a `device("mlx") { }`
block does not hit a device-mismatch error. Contract:
`contracts/eval-contract/clone-model.md`.

### `perturb_params(m, family, sigma, seed)`

Walks `m`'s parameters, filters by `family`, and adds
`sigma * randn(seed, shape)` to each matching parameter in
place. Returns a scalar zero (unit) so the call sits cleanly
in statement or expression position. The i-th affected
parameter uses PRNG seed `seed + i`, so same-shape parameters
within one call get independent deltas while two clones of the
same source with identical `(family, sigma, seed)` produce
bit-identical perturbations. Contract:
`contracts/eval-contract/perturb-params.md`.

### `argtop_k(values, k) -> indices`

Index-returning companion to the existing `top_k(logits, k)`
(which masks logits in place). Returns the `k` indices of the
largest entries in a rank-1 vector, sorted by descending value
with ties broken by lower original index first (stable sort).
Shape `[k]`; integer-valued `f64` entries, matching MLPL's
scalar-as-rank-0 convention throughout. Contract:
`contracts/eval-contract/argtop-k.md`.

### `scatter(buffer, index, value) -> buffer'`

Functional rank-1 scalar write: returns a copy of `buffer`
with the single entry at `index` replaced by `value`. Pairs
with loops (`for i in list { buf = scatter(buf, i, ...) }`)
that produce one number per iteration and accumulate into a
rank-1 accumulator. The source binding is never mutated at
the MLPL source level. Contract:
`contracts/eval-contract/scatter.md`.

## The four families

`perturb_params`'s `family` argument takes one of four strings.
Each maps to a parameter-name filter, with one subtlety: the
"final projection head" is detected **structurally** (the last
top-level `Linear` child of the outermost `Chain`), not by a
name-only heuristic, because `__linear_*` names cannot tell a
transformer MLP layer apart from the vocab projection.

| Family | Touches |
|---|---|
| `all_layers` | Every parameter. |
| `attention_only` | `__attn_Wq_*`, `__attn_Wk_*`, `__attn_Wv_*`, `__attn_Wo_*`. |
| `mlp_only` | `__linear_W_*` / `__linear_b_*`, EXCLUDING the structural head. |
| `embed_and_head` | `__embed_E_*` PLUS the structural head's `W` / `b`. |

Depth-aware families (`early_N_layers`, `late_N_layers`) would
need explicit layer indices in the parameter names, which the
Model DSL does not encode today. Deferred to a follow-up saga;
see the "Not shipped" section below.

## The shipped demo

`demos/neural_thicket.mlpl` is the CPU-only end-to-end:

1. Load the `tiny_shakespeare_snippet` preloaded corpus, train
   a byte-level BPE, window it at context 32.
2. Build a Saga 13-shape Tiny LM (V=280, d=32, h=1) and train
   it for 100 Adam steps.
3. Sweep 4 families x 4 seeds = 16 variants with sigma=0.02,
   score each on the held-out string "to be or not to be that
   is the question" (context 8, because the held-out string is
   short after BPE and the model has no positional encoding so
   shorter evaluation context is fine).
4. `heat = reshape(losses, [4, 4])` and
   `svg(heat, "heatmap")` renders the
   `[family x seed]` specialization heatmap.
5. `best_idx = argtop_k(-1.0 * losses, 4)` picks the K best
   specialists.
6. Ensemble: average logits from all 16 variants into
   `ens_logits` and compute `ens_metric` as the ensemble's
   cross-entropy on the held-out string.

`demos/neural_thicket_mlx.mlpl` is the identical source with
the variant loop + ensemble accumulation wrapped in
`device("mlx") { ... }` (base training stays on CPU). Steps
1-6 are untouched; only the steps 3-6 block moves inside the
MLX scope.

Run (CPU):

```bash
./target/release/mlpl-repl -f demos/neural_thicket.mlpl
```

Run (MLX, Apple Silicon):

```bash
cargo run -p mlpl-repl --features mlx --release -- \
  -f demos/neural_thicket_mlx.mlpl
```

The web REPL at <https://sw-ml-study.github.io/sw-mlpl/>
ships a lesson ("Neural Thickets") running a smaller 4x4
sweep at V=8, d=4 so the heatmap renders quickly in the
browser.

## What the heatmap shows

The four rows are
`all_layers / attention_only / mlp_only / embed_and_head`;
the four columns are seeds 0-3 within that family's seed
range (all_layers starts at 100, attention_only at 200,
mlp_only at 300, embed_and_head at 400, to keep the
per-param seed stride non-overlapping across families).

At `sigma = 0.02` on the Shakespeare base, differences across
families are small but visible: `attention_only` tends to
produce the lowest mean loss (attention projections absorb
small noise well under a short training budget), and
`embed_and_head` the highest (embeddings + output projection
are the most information-dense params per element, so noise
there moves the cross-entropy the most). At `sigma = 0.2` all
four rows dominate by their worst seed and the structure
collapses. At `sigma = 0.001` every row is indistinguishable
from the base.

The headline claim of the Saga 20 demo is NOT
"we found a better model via perturbation" -- at this scale
the base is undertrained and the perturbed losses mostly say
"sigma is big or small". The claim is that the Neural
Thickets workflow is expressible as ~100 lines of MLPL
source, with a first-class specialization heatmap, and that
the four builtins compose cleanly.

## Performance

Measured on an M-class laptop (`cargo bench -p mlpl-bench
--features mlx --bench mlx_vs_cpu -- neural_thicket`):

| Path | Cold | Warm |
|---|---:|---:|
| CPU | 838 us | 767 us |
| MLX | 3.12 ms | 3.01 ms |

MLX is **0.25x** of CPU here, essentially identical to the
Saga 14 Tiny-LM training-step ratio (0.26x). This workload is
inference-only (no autograd, no tape rematerialization), yet
the ratio is the same -- evidence that the bottleneck at
Tiny-LM inner dimensions is per-op kernel launch and
f32 round-trip, not autograd-specific cost. See
`docs/benchmarks.md` for the four-way bottleneck breakdown
from Saga 14; that analysis applies unchanged to this
workload.

The CPU path is authoritative; MLX agrees within fp32
tolerance on every element of the losses vector and the
ensemble logits (see
`crates/mlpl-eval/tests/neural_thicket_mlx_demo_tests.rs`).

## Parity testing

Two integration tests pin the behaviour:

- `crates/mlpl-eval/tests/neural_thicket_tests.rs` (CPU-only,
  any host): runs a cut-down (V=256, d=8, ctx=4, 5 train
  steps, byte-level corpus) end-to-end in under a second,
  asserts losses[16] is finite, heat shape is `[4, 4]`,
  argtop_k returns 4 distinct in-range indices, ensemble
  shape matches a single variant's forward.
- `crates/mlpl-eval/tests/neural_thicket_mlx_demo_tests.rs`
  (triple-gated to macOS + aarch64 + `mlx` feature): runs the
  same cut-down on both paths and asserts losses +
  ensemble logits agree elementwise within fp32 tolerance
  (`1e-3`), heatmap shape matches, and both paths' argtop_k
  returns 4 distinct in-range indices. Exact index equality
  across CPU and MLX is intentionally NOT asserted: losses
  close within tolerance can reorder at the boundary, and the
  workflow cares about ensemble quality, not per-index
  reproducibility.

## Not shipped

Deliberate scope cuts in Saga 20:

- **Depth-aware families (`early_N_layers`, `late_N_layers`).**
  The Model DSL's parameter names encode the *kind* of layer
  (`__linear_*`, `__attn_*`, `__embed_*`) and a monotonic id,
  but not the layer's structural depth. Adding depth-aware
  families needs either a depth-tagged name convention or a
  `ModelSpec`-walking API that returns `(depth, name)` pairs.
  Follow-up saga.

- **Low-rank perturbation.** `perturb_low_rank(m, family,
  rank, seed)` would add `sigma * U @ V.T` where `U`, `V` are
  `randn`-initialized low-rank factors. Implementable on top
  of the family walker that `perturb_params` already uses;
  deferred until someone has a concrete use case.

- **Real pretrained checkpoints.** The demo runs on a from-
  scratch Tiny LM. A Llama-class base needs either a
  checkpoint format (Saga 15+) or an Ollama/LM-server sidecar
  (Saga 19). The perturbation surface itself is checkpoint-
  agnostic and will reuse unchanged.

- **Strict top-K ensembling.** The CPU demo picks
  `best_idx = argtop_k(-losses, 4)` but then averages all 16
  variants' logits rather than rebuilding only the best four.
  Rebuilding requires selecting the family string per flat
  index -- the language does not yet have string-array
  indexing (`families[i / 4]` from the design sketch is not
  expressible). Adding either string-array indexing or an
  index-to-family lookup builtin unblocks strict top-K.

- **Per-iteration counter in `repeat`.** `train N { }` binds
  `step`, but `repeat N { }` does not. The demo uses
  `for i in [0, 1, 2, 3] { }` loops per family instead of the
  design sketch's `repeat 16 { }`. Adding a `step` binding
  to `repeat` is an evaluator-side tweak; not in scope for
  Saga 20.

## Related

- `docs/mlpl-for-neural-thickets.md` -- original design sketch.
- `demos/neural_thicket.mlpl` -- CPU demo.
- `demos/neural_thicket_mlx.mlpl` -- MLX variant.
- `docs/benchmarks.md` -- the MLX bottleneck analysis that
  explains the 0.25x ratio.
- `docs/using-mlx.md` -- MLX reference doc; the device
  scoping + `to_device` surface that the MLX demo uses.
- `contracts/eval-contract/clone-model.md`,
  `perturb-params.md`, `argtop-k.md`, `scatter.md` -- shipped
  behavioural contracts for the four builtins.
