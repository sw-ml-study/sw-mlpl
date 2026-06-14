# Beginner ML Comprehension Plan: make the *process* visible

Status: active (drafted 2026-06-13)
Owner: web-render / web-viz3d / web-paths
Related: `docs/course-outline.md`, `docs/missing-demos.md`,
`docs/language-status.md`, `docs/worker-threads.md`

## Why this plan exists

sw-MLPL is, by its own charter (`docs/course-outline.md`), an ML-teaching
language for absolute beginners: "comfortable with high-school algebra, has
used a programming language, has never trained a neural network." The
educational goal outranks everything else in the project's value ranking
(`docs/optional-typing-design.md`: "educational > correctness > utility ...
>> performance").

A capability audit (49 demos, 11 learning paths, ~19 tutorials, 314 glossary
terms) shows the platform's *breadth* is already excellent: classical ML,
MLPs, RNN/LSTM, attention and transformers from scratch, ViT, GAN, diffusion,
LoRA on real GPUs. The gap is not "more primitives."

The gap is that the platform shows **artifacts** (final tensors, trained
weights, finished loss curves) far better than it shows **processes** (the
training loop, gradient descent, generation step by step). A beginner's core
questions are all about the process:

- "How does ML work?" is a question about the *training loop* and *backprop*.
- "How do I visualize training and inference?" is literally about *process*.
- "What are different approaches (diffusion, generative, hierarchical)?" is
  about *how each family generates*, not a static end result.

Today gradient descent is invisible, the loss curve renders only *after*
training finishes, and inference is shown as a result rather than a sequence.
Closing that is the highest-leverage work for the stated mission.

## What already exists (do not rebuild)

- `train N { body }` captures per-step loss in `last_losses`
  (`components/eval/crates/mlpl-eval/src/eval_blocks.rs`).
- Per-step metric streaming over SSE (`event: metric` frames) on the connect
  path (`components/wasm/crates/mlpl-web-eval/src/eval_sse.rs`).
- Static analysis SVGs: `loss_curve`, `confusion_matrix`, `boundary_2d`,
  `hist`, `scatter_labeled` (`components/viz/crates/mlpl-viz/src/analysis.rs`).
- Live resource sparklines (`TelemetryPanel`) polling `/v1/stats` every 400ms
  (`components/web-render/crates/mlpl-web-render-aux/src/telemetry_panel.rs`).
- 3D "sculpture" stage with attention heatmaps, Sankey model graphs, and a
  clickable derivation trace (`components/web/crates/mlpl-web/js/stage3d.js`).
- `grad`, `adam`, `momentum_sgd`, LR schedules
  (`components/eval/crates/mlpl-eval/src/grad.rs`, `grad_optim.rs`).

The data for "show the process" is largely already flowing. Most of this work
is rendering and narrative, not new ML infrastructure.

## Known constraint: single-threaded WASM blocks during local eval

In the in-browser (non-connect) path, `train N {}` runs synchronously in
WASM and blocks the UI thread (see `docs/worker-threads.md` and the
demo-UI-block note). True per-step live animation requires either the
streaming connect path (server runs training, streams metric frames) or
**chunked** local training (`train 5` repeated, repainting between chunks --
the demos already do this for progress notes). The plan below targets the
CPU/browser path with chunked repaint as the baseline so it works in the
public live demo, and uses the SSE stream for smoother updates when connected.

## The four work items (do in order)

### 1. This plan (done when committed)

Capture the gap analysis and the roadmap below as a durable, reviewable
artifact before building. This file.

### 2. Live training visualization

Goal: a beginner watches the loss *descend as training happens*, and sees
overfitting as a *widening gap*, instead of staring at "Evaluating..." and
getting a finished curve.

Deliverables:
- A live loss chart that updates while a `train` block runs (per-chunk on the
  local path, per-`metric`-frame on the connect path).
- Train vs validation loss on the same axes (val via the existing
  `val_split` / held-out loss builtins) so overfitting is visible.
- A demo ("Watch a model learn") wired to use it, plus a glossary/tour pointer.

Acceptance:
- Running the demo shows the curve growing left-to-right during training.
- The train/val gap is visibly demonstrated on an intentionally over-capacity
  model.
- Works on the public CPU/browser demo (chunked), smoother when connected.

Primary files: `components/viz/crates/mlpl-viz/src/analysis.rs`,
`components/web-render/`, `components/wasm/crates/mlpl-web-eval/src/eval_sse.rs`,
a new/updated demo in `components/web-demos/`.

### 3. "How ML works" visualization (gradient descent + backprop)

Goal: make the mechanism of learning visible -- the single biggest
conceptual hole.

Deliverables:
- A loss-landscape view: sweep two weights, render the loss surface
  (reuse heatmap), and walk the optimizer's trajectory across it step by step
  so "gradient descent" is a path downhill, not a formula.
- A gradient-flow view: per-layer gradient magnitudes for one backward pass,
  so "backprop" is layers lighting up by how much they learn.
- A demo ("How gradient descent works") and glossary/tour wiring.

Acceptance:
- The trajectory visibly rolls downhill on the loss surface; changing the
  learning rate visibly changes the path (overshoot vs creep).
- Per-layer gradient magnitudes render for a small MLP.

Primary files: `components/viz/crates/mlpl-viz/src/`,
`components/eval/crates/mlpl-eval/src/grad*.rs` (to surface per-layer grad
norms), a new demo in `components/web-demos/`.

### 4. The beginner spine: "How does ML work? (start here)"

Goal: one narrative on-ramp that sequences existing demos plus the new
process-visualizations into a single guided story, so a newcomer asking "how
does ML work?" has one obvious place to begin.

Deliverables:
- A new learning path `PATH_HOW_DOES_ML_WORK__START_HERE` ordering: a tensor/
  data warm-up -> a single neuron + loss -> gradient descent viz (item 3) ->
  watch-it-learn (item 2) -> a tiny network -> generation step by step.
- A short "families of approaches" lesson contrasting the existing
  discriminative, autoencoder, GAN, and diffusion demos under one lens
  ("how does each one generate?").

Acceptance:
- The path appears in the paths list and steps run in sequence.
- Each step links the relevant glossary terms and the two new demos.

Primary files:
`components/web-paths/crates/mlpl-web-paths-data/src/`,
`components/web-tutorial/`, `docs/glossary.md`.

## Sequencing and status

- [x] 1. Plan (this doc)
- [x] 2. Live training visualization -- `train_val_curve` builtin + the
      "Watch a Model Learn (overfitting)" demo (train vs validation on shared
      axes, rendered at the halfway mark and the end). A true in-place
      live-loss panel over the SSE stream remains a connect-path follow-up.
- [x] 3. Gradient-descent / backprop visualization -- `loss_landscape`
      builtin (a 2-weight loss surface heatmap with the optimizer trajectory
      walked across it) + the "How Gradient Descent Works" demo, which fits
      y = w*x + b, draws the whole loss surface, walks gradient descent
      downhill on it, and shows the loss falling and the gradient magnitude
      shrinking as the path flattens.
- [x] 4. Beginner spine path + generative-families lesson -- two new learning
      paths in `mlpl-web-paths-data`: "How does ML work? (start here)"
      sequences Hello Numbers -> Basics -> How Gradient Descent Works -> Watch
      a Model Learn -> XOR-MLP -> Tiny LM Generate with glossary anchors; "How
      models generate" contrasts the autoencoder / GAN / diffusion families on
      toy 2D data. Both registered in the paths aggregator. (A path-reference
      resolution test is a worthwhile follow-up -- bad refs currently fall
      back silently in the UI.)

All four plan items are done. Follow-ups also landed: a PATHS
reference-resolution test (`path_refs_resolve.rs` -- every `Step::Demo` /
`Step::Glossary` must resolve, so a mistyped reference fails the build
instead of silently falling back in the UI) and a warning-paydown pass.
Final sw-checklist: 18 FAIL / 330 WARN (both below the 20 / 331 baseline).

Each item ships as its own commit (or small commit series), holds the
sw-checklist FAIL/WARN counts flat for our code (`sw-checklist components/`),
rebuilds `pages/` when web source changes, and updates this checklist. Genuine
ML-primitive gaps (batch-norm, dropout, weight-decay, GRU, nucleus/beam
sampling, `kl_div` for distillation, checkpoint save/load) are tracked
separately in `docs/missing-demos.md` and the course outline; they matter less
for beginner *comprehension* than the process-visualization work above and are
deliberately out of scope for this plan.
