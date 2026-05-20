# Better cat-vs-dog: future-demo recommendations

Status: design doc, written 2026-05-19 alongside Saga 29 step 017
Driving question: the in-browser pets ViT classifies trained
images 100% correctly but misclassifies almost every held-out
photo as a cat. What does the next saga of demos look like?

## What ships today

Three trained-classifier demos in `apps/mlpl-web/src/demos.rs`:

| Demo | Train set | Steps | Heads | Validation? |
|---|---:|---:|---:|---|
| Pets: cat vs dog (quick) | 8 (4+4) | 30 | 1 | No (training-set acc) |
| Pets: predict + gallery | 16 (8+8) | 30 | 1 | No |
| Pets: multi-head ViT (quick + viz) | 8 (4+4) | 30 | 4 | No |

Plus the CLI-only `demos/vit_multihead_thorough.mlpl` (200
adam steps, 20 images, 4 heads, `device("mlx")` scope).

All four are **deliberately overfit**. The browser variants
have to finish in WASM in under a minute, so training set
size and step count both stay tiny. Training accuracy
reaches 1.0 in every case. None of them measure
generalization.

## Why generalization fails today

In order of effect on held-out accuracy:

1. **Training set is too small** (8 to 20 images). Cat-vs-dog
   has enormous within-class variance; a model trained on 4
   cat photos has memorized those 4 cats, not "cat-ness".
2. **No train/val split.** The "accuracy" reported is on the
   training set. Memorization is indistinguishable from
   learning from this number.
3. **No regularization.** No weight decay (MLPL ships Adam,
   not AdamW), no dropout (not a builtin), no early
   stopping. The model trains until cross-entropy is
   effectively zero on the training set, which by
   construction kills generalization.
4. **No data augmentation.** Random horizontal flip alone
   doubles effective dataset size for free; random crop +
   color jitter would multiply it further. None ship as
   builtins.
5. **Resolution is small** (64x64). Texture cues that
   distinguish breeds get downsampled away. The 128x128
   `fetch_dataset` path exists but the WASM bundle doesn't
   ship it.
6. **Architecture is minimal** (1 transformer block, no
   LayerNorm with learned affine, ReLU instead of GELU).
   Tier 2 builtins (layer_norm, gelu_layer) from
   `docs/milestone-vit.md` Phase 3 are deferred; once
   they land, 2-4 stacked blocks become viable.
7. **No pretrained init.** All weights start from `randn`.
   Modern ViTs are pretrained on ImageNet-21k or similar;
   MLPL doesn't ship pretrained checkpoints and there's no
   load_model / save_model builtin yet.

## Recommended demo ladder

Six new demos / variants, ranked by leverage-per-step. Each
fits in one saga step.

### 1. Full-pets_tiny training + held-out validation (highest leverage, smallest step)

**Demo name:** `Pets: full set + validation split`.
**Builtins needed:** none (all already ship).
**Effort:** small. ~50 new lines in `demos.rs`; one new
integration test.

Train on 140 images (70 cat + 70 dog) of `pets_tiny`, hold
out the remaining 60 (30 cat + 30 dog), report **val
accuracy** at the end instead of training accuracy. This is
the single biggest jump: training memory goes from "4 cats"
to "70 cats", and the val number is honest. Expect
generalization to leap from "always cat" to "~70-80%
correct on held-out", roughly matching the saga's plan-doc
expectation.

WASM constraint: 140 images x 100 steps with the current
interpreter takes ~60s in release. Acceptable -- this
becomes the demo that shows training takes real time.

### 2. Confidence-thresholded "other" output (no new training, post-processing)

**Demo name:** `Pets: with-confidence classification`.
**Builtins needed:** `reduce_max(softmax(logits), 1)` --
all ship.
**Effort:** small. Tail block on existing demo + one
threshold sweep viz.

Take the trained classifier (any variant), compute softmax
probabilities, treat any prediction with max-probability
below a threshold as "uncertain / other". This catches the
"non-pet image gets confidently classified" failure mode
cheaply -- no new training data, no new arch. The right
threshold needs a calibration plot; emit it as a sweep
chart so the user sees the trade-off (low threshold = high
recall, high threshold = high precision).

### 3. Horizontal-flip augmentation (smallest new builtin)

**Demo name:** `Pets: with augmentation`.
**Builtins needed:** new `flip(x, axis)` builtin
(differentiable; tape-pure rearrangement).
**Effort:** medium. New tape NodeKind + backward + runtime
dispatch + tape lowering + tests, plus the demo.

Pre-compute a flipped copy of every training image,
concatenate to the batch, train on the doubled set.
Doubling the dataset for free is the cheapest single jump
in generalization. Random flip (50% probability per image
per step) is the canonical version; static double-batch
is the simpler-to-implement variant and gets most of the
benefit.

### 4. AdamW (weight decay)

**Demo name:** wired into existing demos as a swap.
**Builtins needed:** new `adamw(loss, params, lr, beta1,
beta2, eps, weight_decay)` builtin.
**Effort:** medium. Adam is already implemented as a
state-tracked optimizer; AdamW is a 4-line tweak (subtract
`lr * weight_decay * param` after the update step). Tape
lowering is identical to Adam.

Weight decay is the standard regularizer for transformers.
With weight_decay=0.01 and a small enough lr, this alone
typically claws back 5-10 percentage points of val
accuracy on a small dataset.

### 5. LayerNorm with learned affine + GELU (Tier 2 builtins)

**Demo name:** `Pets: thorough multi-head (real LN + GELU)`.
**Builtins needed:** `layer_norm(d, seed)` DSL layer (mean-
center + unit-variance + learned `[d]` gamma + beta),
`gelu(x)` builtin + `gelu_layer()` DSL wrapper (tanh
approximation).
**Effort:** medium-large. Both have well-defined backwards
already documented in `docs/milestone-vit.md` Phase 3 (deferred).

Substituting `rms_norm` -> `layer_norm` and `relu_layer` ->
`gelu_layer` brings the architecture into actual upstream
ViT parity. Worth doing both as one step because the
gradcheck and tape lowering machinery overlaps.

### 6. Real 3-way classifier (cat / dog / other)

**Demo name:** `Pets vs other: 3-way classifier`.
**Builtins needed:** none architecturally (change
`linear(_, 2, ...)` to `linear(_, 3, ...)`). **Dataset
needed:** an "other" image source -- can be CIFAR-100 with
cat+dog removed, or synthetic noise / solid colors / Pets-
v-Pascal-VOC negatives.
**Effort:** large -- mostly data sourcing. The "other"
class is near-infinite and class imbalance bites; you need
a balanced sampling scheme and probably more steps to
converge.

Practically: do #2 (confidence threshold) first as the
cheap shortcut, treat #6 as a longer-term option if a
demand emerges for explicit "I don't know" outputs.

### 7. Larger model + higher resolution + fetch_dataset

**Demo name:** CLI-only; bigger version of
`vit_multihead_thorough.mlpl`.
**Builtins needed:** the Tier 2 set from #5, plus
`fetch_dataset("oxford_iiit_pet")` (already ships) at
128x128.
**Effort:** large. Real training run -- minutes on CPU,
seconds-to-minutes on MLX. Probably the right place to
finally show off the `device("mlx") { ... }` scope at
non-trivial work size.

This is the "do everything right" baseline. ~80-90% val
accuracy on Oxford-IIIT Pet is realistic; the upstream
paper gets there at this scale.

## Suggested ordering

If the demo ladder ships one step at a time, the maximum
educational payoff per step:

1. **#1 full-set + validation split** -- changes the "always
   cat" failure overnight, shows the user what real
   generalization looks like, exposes the train/val gap
   that drives every following step.
2. **#2 confidence threshold** -- directly addresses the
   "non-pet image gets misclassified" concern, no new
   training cost.
3. **#3 horizontal flip** -- introduces augmentation as a
   pattern; the new `flip` builtin compounds with future
   demos.
4. **#4 AdamW** -- generic optimizer improvement that
   benefits every downstream demo.
5. **#5 LayerNorm + GELU** -- closes the architectural gap
   to the upstream notebook.
6. **#7 thorough on MLX** -- the headline "ViT trained
   properly" demo. Apple-specific (per the Arch-move
   planning), so it's also the deadline-driven step before
   the dev-host migration.
7. **#6 3-way classifier** -- ships only if there's user
   demand for explicit "other" output beyond what #2
   buys.

## Non-recommendations

Things that look tempting but aren't worth the complexity
at this scale:

- **Bigger model (more heads, bigger d_model).** With only
  200 training images, capacity isn't the bottleneck --
  data is. A bigger model overfits faster, not better.
- **Learning-rate schedules** (cosine, warmup). MLPL ships
  these but the win on a 200-image dataset trained for
  ~100 steps is in the noise. Worth turning on for #7
  only.
- **Stacked transformer blocks** (depth 2+). Same as
  bigger-model: 200 images doesn't support more capacity.
  Add depth only after #7's fetch_dataset run.
- **Custom losses (focal, label-smoothing).** Class
  imbalance isn't the failure mode here; data scarcity
  is. Cross-entropy is fine.
- **Distillation, contrastive pretraining, masked-image
  modeling.** All cool but architectural changes beyond
  the current scope of MLPL's Model DSL.

## Cross-reference: deferred-primitives queue

The builtins each demo needs are tracked in
`docs/plan.md`'s "Deferred primitives queue" section. The
demos here pull from that queue in roughly the order above:

| Demo | Pulls from queue |
|------|---|
| #1 full set + val | -- (uses what ships) |
| #2 confidence threshold | -- (uses what ships) |
| #3 horizontal flip | `flip` (new entry) |
| #4 AdamW | `adamw` (new entry) |
| #5 LayerNorm + GELU | `layer_norm`, `gelu`, `gelu_layer` (queued) |
| #6 3-way | -- (data work, no builtins) |
| #7 thorough on MLX | LayerNorm + GELU + dataset already-shipped |

When any of these graduate from "queued" to "in-progress"
status, the corresponding entry here should pick up its
saga step number.
