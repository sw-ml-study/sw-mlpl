# Vision Transformer Milestone (Saga 29)

## Why this exists

MLPL has trained a tiny language model end-to-end since Saga 13.
The next obvious "real architecture" demo is vision: a Vision
Transformer on the Oxford-IIIT Pet dataset, discriminating cats
from dogs. The reference implementation is two notebooks
(`Basic_Vision_Transformer_(ViT).ipynb`,
`Image_balanced_Basic_Vision_Transformer_(ViT).ipynb`) in the
sibling `Vision-Transformer-ViT` checkout; the demo plan derived
from those notebooks is `docs/ViT-demo-plan.md`.

ViT is the right next demo for three reasons:

- **It is structurally close to what we already ship.** The Saga
  11 Model DSL and the Saga 13 transformer block cover most of
  the architecture; what is missing is a small, named set of
  primitives (`load_images`, `patchify`, `concat`,
  `layer_norm`, `gelu`, multi-head autograd).
- **It is a new modality.** Sagas 6-13 cover scalars, vectors,
  matrices, embeddings, and text. Images add a new shape regime
  (`[B, C, H, W]`), a new I/O path (PNG/JPEG decode), and a new
  visualization (attention-over-patches over an image).
- **It motivates the deferred Saga 21 follow-ups.** The thorough
  ViT variant wants to train on the MLX peer and have its loss
  curve stream into the browser REPL. That is exactly the surface
  Saga 21.5 unlocks, which makes Saga 29 the headline use case
  that proves out the multi-client UI work.

Goal ranking applied:

- **Educational** leads: the ladder runs from a no-training
  attention-pattern viz up to a thorough trained model, each step
  introducing one new concept.
- **Correctness**: parity vs the upstream PyTorch notebook for a
  fixed seed and identical hyperparameters anchors the
  implementation.
- **Utility**: the demos are runnable from REPL, compiled app,
  and (after Saga 21.5) browser-against-MLX-peer.
- **Performance** is explicitly last; this is not a speed claim.

## Non-goals

- **Full-scale ViT.** Target is the upstream "minimal ViT"
  hyperparameters: 128x128 input, 16x16 patches, d=128, MLP=256,
  one transformer block. Stacked blocks and full ViT-B / ViT-L
  configurations are out.
- **Pretraining claims.** Train-from-scratch on a tiny dataset
  only. No checkpoints from ImageNet.
- **Other image datasets.** Oxford-IIIT Pet only. CIFAR / MNIST /
  ImageNet variants land later if user demand surfaces.
- **CUDA dispatch.** ViT runs on CPU and (via Saga R1) on the MLX
  peer. CUDA waits for R2.
- **Object detection / segmentation.** Classification only.
- **Data augmentation.** Resize + normalize only. RandomCrop /
  Flip / ColorJitter are good follow-ups; ship later.

## Dependencies

- **Saga 21.5 (Multi-client UI follow-up)** delivers Phase 4
  web-rerouting + Phase 6 wire dtypes. Demo 3 (thorough
  multi-head, in the browser, training on MLX) needs both.
  Demos 1 and 2 do not -- they run fully in WASM.
- **Saga R1 (MLX as a service)** is shipped. The thorough demo
  uses `device("mlx") { ... }` and the existing peer protocol.
- **Saga 23 (Typed ML values)** is shipped. The demos inherit
  typed tags on `Logit`, `Probability`, `Loss`, `Gradient`,
  `Weight`, `AttentionMap` for free.

## Quality requirements (every step)

Identical to Saga 23. TDD, four `cargo` gates +
`markdown-checker` + `sw-checklist` green, `/mw-cp` checkpoint,
push after every commit. Web UI changes rebuild `pages/`.
`.agentrail/` committed.

New-builtin steps also have a gradcheck-parity requirement: every
new differentiable op ships with a finite-difference parity test
on a CPU fixture and an MLX fixture (within fp32 tolerance), the
same standard Saga 14 set for the MLX backend.

## What already exists

- Model DSL: `linear`, `chain`, `residual`, `rms_norm`,
  `attention(d, h, seed)` (forward path), `relu_layer`,
  `tanh_layer`, `softmax_layer`, `embed`, `sinusoidal_encoding`,
  `cross_entropy` (Sagas 11, 13).
- `apply(model, X)` is differentiable on the tape for the layers
  above with `h = 1` (Saga 11 step 005b).
- `train { }`, `experiment { }`, `adam`, `cosine_schedule`,
  `linear_warmup`, `last_losses`, `loss_curve` (Saga 10).
- `randn(seed, shape)`, `reshape`, `transpose`, `matmul`,
  `softmax`, `reduce_*` (Sagas 6, 8).
- `svg(matrix, "heatmap")` (Saga 7).
- `load_preloaded("<name>")` precedent for embedded datasets
  (Saga 12). Today only `tiny_shakespeare_snippet` ships.
- `device("mlx") { ... }` + `to_device("cpu", x)` + peer
  routing (R1).
- Typed tags propagate through every op the demos use (Saga 23).

## What is missing

Eight gaps, grouped by tier per `docs/ViT-demo-plan.md`:

**Tier 1 (required for the single-head, quick-training demo):**

1. `load_images(dir, [H, W]) -> [N, C, H, W]` builtin.
2. `load_preloaded("pets_tiny")` embedded fixture (~200 images,
   bundled in the WASM target so the browser tutorial works).
3. `fetch_dataset("oxford_iiit_pet")` builtin (native-only;
   WASM falls back to the preloaded fixture with a warning).
4. `patchify(x, P)` builtin, differentiable on the tape.
5. `concat(a, b, axis)` builtin, differentiable on the tape.

**Tier 2 (required for the thorough multi-head demo):**

6. `gelu(x)` builtin + `gelu_layer()` DSL layer (tanh
   approximation; gradcheck parity).
7. `layer_norm(d, seed)` DSL layer with learned affine.
8. Multi-head attention on the tape: `attention(d, h, seed)`
   currently rejects `h > 1` in `crates/mlpl-eval/src/model_tape.rs`;
   lower the per-head reshape / transpose / per-head matmul /
   recombine onto the tape.

**Tier 3 (ergonomic, not blocking):**

- u8/f32 dtype on the MLX peer wire -- moved to Saga 21.5
  Phase 6, listed here only for cross-reference.
- `save_model` / `load_model` -- Saga 15 follow-up; demo 4
  (attention readout from a trained checkpoint) is gated on it.
- `attention_map_over_patches(model, image)` convenience helper
  -- nice-to-have, can wait.

## Phases

### Phase 1 -- Tier 1 builtins (3 steps)

#### Step 001 -- load_images + pets_tiny preloaded fixture
New `load_images(dir, [H, W])` builtin in `crates/mlpl-runtime`:
decode PNG / JPEG via `image-rs`, resize to `[H, W]`, normalize
to f64 in `[-1, 1]`. Returns `[N, 3, H, W]` with `[batch,
channel, y, x]` labels. Native-only at this step; WASM raises a
clean error pointing at `load_preloaded`.

In parallel, build the `pets_tiny` fixture: 100 cat + 100 dog
images from Oxford-IIIT Pet, resized to 64x64 (smaller than the
demo target to fit in the WASM bundle), normalized, serialized
as a single `.bin` blob shipped alongside
`tiny_shakespeare_snippet`. `load_preloaded("pets_tiny")` returns
`{X: [200, 3, 64, 64], Y: [200], names: [str]}`.

Tests: decode known PNG fixture matches a recorded byte hash
(within fp tolerance), WASM build does not pull in `image-rs`
(feature-gated), `load_preloaded("pets_tiny")` round-trip
verifies one labelled cat and one labelled dog.

#### Step 002 -- fetch_dataset
`fetch_dataset("oxford_iiit_pet")` downloads the tarball to
`$MLPL_DATA_DIR` on first use, verifies sha256, extracts,
decodes via the Step 001 builtin, returns the same shape as
`load_preloaded("pets_tiny")` but at the demo's 128x128
resolution. Native-only; WASM raises a clean error pointing at
the preloaded fixture.

Tests: a recorded-fixture small dataset (NOT the real Oxford-IIIT
Pet -- a 4-image synthetic tarball) drives the test; the live
tarball download is exercised by an ignored test run with
`--ignored`.

#### Step 003 -- patchify + concat with tape lowering
`patchify(x, P)` over `[B, C, H, W]` returns `[B, N, P*P*C]`
where `N = (H/P) * (W/P)`. Internally implemented as reshape +
transpose + reshape; the named builtin exists so the tape has
one op to lower and the demo reads cleanly.

`concat(a, b, axis)` produces an array whose shape matches both
inputs except along `axis`, where the sizes add. Initial version
supports two-argument concat over axis 0 or 1; variadic and
arbitrary axis are follow-ups.

Both ops on the tape: forward identical to eager, backward
splits the gradient along the same axis they joined. Gradcheck
parity vs finite differences on `[2, 3, 4, 4]` patchify and
`[2, 1, 4]`-prepended-to-`[2, 8, 4]` concat fixtures.

### Phase 2 -- single-head ViT demo (2 steps)

#### Step 004 -- demos/vit_attention_pattern.mlpl
No training. Decode one image from `pets_tiny`, patchify,
linear-embed, prepend a randn CLS token, add a randn positional
embedding, run one `attention(128, 1, ...)` head, render the
attention matrix with `svg(..., "heatmap")`. Exercises every
Tier 1 builtin end-to-end.

Tests: row sums of the rendered attention matrix are within
`1e-6` of 1.0; the SVG renders to a non-empty file; the same
seed produces the same matrix.

#### Step 005 -- demos/vit_single_head_quick.mlpl
Full training loop matching `docs/ViT-demo-plan.md` Demo 2.
CPU-only. Single-head attention, `rms_norm`, `relu_layer`.
Trains 100 steps on the preloaded fixture, validation accuracy
beats the 60% threshold for a fixed seed.

The web REPL tutorial gains a "Vision Transformers" lesson
that runs this demo verbatim. The demo dropdown gets a "Pets:
cat vs dog (quick)" entry.

Tests: integration test in `crates/mlpl-eval/tests/` runs the
demo to completion and asserts the val-accuracy threshold;
WASM build runs the same source.

### Phase 3 -- Tier 2 builtins (3 steps)

#### Step 006 -- gelu builtin + gelu_layer
`gelu(x)` as a scalar builtin (tanh approximation: `0.5 * x *
(1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`), `gelu_layer()`
as the Model DSL wrapper. Tape lowering matches the
`relu_layer` precedent; gradcheck parity on `[2, 5]` fixture.

#### Step 007 -- layer_norm with learned affine
`layer_norm(d, seed)` DSL layer: mean-center, unit-variance,
elementwise `gamma * x + beta` with learned `[d]`-shaped gamma
and beta. Tape lowering parallel to `rms_norm`. Gradcheck
parity on `[2, 6, 4]` fixture for both data input and the
gamma/beta parameters.

#### Step 008 -- multi-head attention on the tape
`crates/mlpl-eval/src/model_tape.rs` currently rejects `h > 1`.
Replace the rejection with a per-head lowering: reshape `[B, T,
d]` to `[B, T, h, d/h]`, transpose to `[B, h, T, d/h]`,
per-head scaled-dot-product softmax + matmul, transpose back,
reshape, output projection.

Backward: every primitive in the lowering already has a tape
rule, so the multi-head backward is "the same backward, but with
the head axis present everywhere." Gradcheck parity vs the
single-head path on a `d=16, h=4` fixture (where the multi-head
output equals a particular single-head computation if the head
weights are chosen carefully -- the standard gradient sanity
check for multi-head implementations).

### Phase 4 -- thorough ViT demo (2 steps)

#### Step 009 -- demos/vit_multihead_thorough.mlpl
Full training loop matching `docs/ViT-demo-plan.md` Demo 3. Uses
`fetch_dataset("oxford_iiit_pet")`, `layer_norm`, `gelu_layer`,
`attention(128, 4, ...)`, wrapped in `device("mlx") { ... }`.
30 epochs.

Tests: integration test runs one epoch on CPU and MLX with a
fixed seed and asserts identical-within-tolerance loss; longer
multi-epoch runs gated by `--ignored`. Final accuracy threshold
documented in `docs/benchmarks.md`.

#### Step 010 -- demos/vit_attention_viz.mlpl
Gated on `save_model` / `load_model` shipping. If those are not
in place, this step ships as a tail block inside Demo 3 instead
of a standalone program: extract the CLS row of the attention
matrix on a held-out test image, reshape over the 8x8 patch
grid, render as a heatmap overlay on the image.

### Phase 4.5 -- Predictions UI (3 steps)

These three lift the ViT track from "loss curve + final
accuracy" into "look at the model's decisions on actual
photos" -- the canonical user request. Each step is small and
can land in any order after Phase 4's `demos/vit_multihead_thorough.mlpl`
trains a usable checkpoint.

#### Step 010a -- gallery viz output
New `svg(images, "gallery")` viz output (or `gallery(images,
labels=None, predictions=None) -> string`): take an
`[N, 3, H, W]` image tensor, emit an SVG/HTML grid of N
thumbnails laid out NxM with optional label / prediction
overlay strings. Reuses the Saga 21.5 step 005 viz-format
table for the right `Content-Type` so the result renders
inline in the web REPL accordion. CPU-only, no training.
Test fixture: render the 16-image `pets_tiny[..16]` slice
with labels.

#### Step 010b -- predict_batch builtin + labeled gallery demo
`predict_batch(model, X) -> Y` (CPU + MLX) runs the trained
classifier head over a batch of inputs, returns the argmax
labels. New `demos/vit_predict_gallery.mlpl`: trains (or
loads) a ViT, predicts the validation slice, renders the
gallery annotated with `actual: cat / predicted: dog` (or
similar) so misclassifications stand out at a glance. The
demo runs against the same `pets_tiny` fixture so it works
in the WASM REPL once Demo 2 ships.

#### Step 010c -- bring-your-own-image
Two paths share the same plumbing:
- CLI: new `load_image(path)` builtin that decodes a single
  JPG/PNG to a `[3, H, W]` u8 tensor (which the Saga 21.5
  step 011 u8 wire dtype carries to the MLX peer
  unchanged).
- Web: a small file-picker UI in the connect-mode REPL
  (`apps/mlpl-web/src/handlers.rs`) lifts a user-selected
  image into bytes -> POSTs to a new
  `/v1/sessions/<id>/upload-image` endpoint on `mlpl-serve`
  -> server-side decodes + binds as a u8 array under a
  caller-chosen variable name.

Both paths feed the same `apply(model, X)` + gallery viz
pipeline so the user's image lands beside the held-out
pets_tiny samples with the same labeled-prediction overlay.

### Phase 5 -- Tutorial + bundle + release (2 steps)

#### Step 011 -- web bundle + tutorial
New "Vision Transformers" tutorial lesson with the Demo 1
attention-pattern walkthrough, the Demo 2 training run, the
Phase 4.5 predictions-gallery demo, and a short forward-look
at the thorough demo. Demo dropdown gains all the ViT
entries (Demo 4 prints "needs save_model" if checkpointing
isn't shipped yet). Rebuild `pages/` via
`scripts/build-pages.sh` and commit both source and built
artifact in the same commit.

#### Step 012 -- release v0.21.0
Bump workspace version. `docs/using-vit.md` retrospective + user
guide. Update `docs/saga.md`, `docs/status.md`,
`docs/are-we-driven-yet.md`. Tag `v0.21.0`. Push commit and
tag.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | load-images-and-pets-tiny    | 1 | `load_images` + `load_preloaded("pets_tiny")` |
| 002 | fetch-dataset                | 1 | `fetch_dataset("oxford_iiit_pet")` |
| 003 | patchify-and-concat          | 1 | `patchify` + `concat` with tape lowering |
| 004 | vit-attention-pattern-demo   | 2 | `demos/vit_attention_pattern.mlpl` |
| 005 | vit-single-head-quick-demo   | 2 | `demos/vit_single_head_quick.mlpl` + tutorial |
| 006 | gelu                         | 3 | `gelu` builtin + `gelu_layer` |
| 007 | layer-norm                   | 3 | `layer_norm(d, seed)` with learned affine |
| 008 | multi-head-attention-tape    | 3 | `attention(d, h, seed)` differentiable for `h > 1` |
| 009 | vit-multihead-thorough-demo  | 4 | `demos/vit_multihead_thorough.mlpl` on MLX |
| 010 | vit-attention-viz-demo       | 4 | `demos/vit_attention_viz.mlpl` (or inline) |
| 010a | gallery-viz                 | 4.5 | `svg(images, "gallery")` viz output |
| 010b | predict-batch-and-gallery-demo | 4.5 | `predict_batch` + `demos/vit_predict_gallery.mlpl` |
| 010c | load-image-and-upload-endpoint | 4.5 | CLI `load_image(path)` + web file-picker -> `/v1/sessions/<id>/upload-image` |
| 011 | vit-tutorial-and-bundle      | 5 | tutorial lesson + demo dropdown + pages rebuild |
| 012 | release-v021                 | 5 | version bump, retrospective doc, tag |

Fifteen steps. Steps 001-005 (Tier 1 + single-head demo) are the
"can land before Saga 21.5" subset; everything from 008 onward
benefits from the f32/u8 wire dtype landing first, and Demo 3 in
the browser requires the web-rerouting from Saga 21.5.

## Success criteria

- `load_preloaded("pets_tiny")` returns `[200, 3, 64, 64]` with
  `[batch, channel, y, x]` labels and 100 cats + 100 dogs.
- `patchify(randn(0, [1, 3, 128, 128]), 16)` returns shape
  `[1, 64, 768]` and gradchecks against finite differences.
- `concat(randn(0, [1, 1, 4]), randn(1, [1, 8, 4]), 1)` returns
  shape `[1, 9, 4]` and the backward splits cleanly.
- `gelu(0.0)` equals 0; `gelu` matches PyTorch's
  `nn.GELU(approximate='tanh')` to within `1e-6` on a fixture.
- `layer_norm(4, 0)` applied to a constant input returns zeros
  before the affine; the learned affine reproduces the input
  when `gamma=1, beta=0`.
- `attention(128, 4, 0)` inside `grad(...)` returns gradient
  shapes matching the parameter shapes (no `h > 1 unsupported`
  error).
- `demos/vit_attention_pattern.mlpl` renders an attention matrix
  whose row sums are within `1e-6` of 1.0.
- `demos/vit_single_head_quick.mlpl` beats 60% validation
  accuracy in 100 train steps for a fixed seed.
- `demos/vit_multihead_thorough.mlpl` produces identical (within
  fp32 tolerance) per-epoch loss on CPU and MLX for a fixed seed.
- The web REPL tutorial "Vision Transformers" lesson runs end-to-
  end via the existing "Run All" path.
- All quality gates green; pages deployed; release tagged.

## Risks and open questions

- **WASM image bundle size.** 200 cat/dog images at 64x64x3 f32
  is ~10 MB in the WASM blob. Compress at build time
  (deflate-then-base64 in the embedded constant), decompress on
  first use. If 10 MB is too heavy, ship the fixture as a
  separate `pets_tiny.bin` URL that the WASM client `fetch`es on
  first use -- same content-addressed scheme as the viz cache.
- **PNG / JPEG decode in WASM.** Step 001 keeps `image-rs` out of
  the WASM target; the embedded fixture is already-decoded floats.
  `fetch_dataset` is native-only by design. Re-evaluate after the
  Saga 21.5 web-rerouting work lands -- in connect mode the
  decode can happen on the server.
- **Color channel order.** Pin to RGB `[B, C, H, W]` to match the
  upstream notebook. Document in the `load_images` builtin
  docstring; never assume the user matches.
- **Multi-head tape lowering correctness.** The standard sanity
  check is "single head with a specific weight choice equals
  multi-head with `h=1`." Implement that as the first test; the
  general `h>1` gradcheck is the second.
- **MLX memory footprint.** Full balanced Oxford-IIIT Pet at f64
  on the peer is ~3.5 GB. The thorough demo wants batch-level
  forwarding (one batch per `device("mlx") { ... }` eval) rather
  than ship-once-train-many. This may force a batch-streaming
  pattern earlier than planned; if so, lift it out as Saga 21.6
  rather than ballooning Saga 29.
- **upstream comparison.** "Within the band the upstream notebook
  reports" needs a concrete acceptance number. Run the upstream
  notebook 5 times for each variant, record min / mean / max
  validation accuracy, and set the MLPL acceptance threshold at
  the lower bound. Do this *before* writing Step 009's test
  assertion.

## References

- `docs/ViT-demo-plan.md` -- the demo-companion plan with the
  four-demo ladder and code sketches.
- `docs/milestone-multi-client-followup.md` -- Saga 21.5;
  delivers web-rerouting and wider wire dtypes.
- `docs/milestone-mlx-backend.md` -- Saga 14; precedent for
  device-scoped training and gradcheck parity tests.
- `docs/milestone-tiny-lm.md` -- Saga 13; precedent for an
  end-to-end model demo with tutorial integration.
- Upstream reference notebooks: sibling checkout at
  `../Vision-Transformer-ViT/`.
