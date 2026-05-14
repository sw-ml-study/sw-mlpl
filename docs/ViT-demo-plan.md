# ViT demo plan

## Goal

Build a small Vision Transformer track for sw-MLPL: image patch
embedding, learnable CLS token, learnable positional embedding, one
transformer block (LayerNorm -> attention -> residual -> LayerNorm ->
MLP -> residual), and a linear classification head on the CLS row,
trained to discriminate cats from dogs on the Oxford-IIIT Pet dataset.

The demo should not chase published ViT accuracy numbers. The goal is
to show that one auditable `.mlpl` source file can express the whole
experiment -- dataset, model, training loop, attention visualization,
and metrics -- across the same primitives the language already uses
for tiny language models. The ladder runs from a no-training
attention-pattern visualization up to a multi-head model trained on
the MLX peer.

## Why this is better than a notebook

- The upstream reference is two Colab notebooks (basic and balanced)
  whose cell order matters and whose hyperparameters live in opaque
  cell-scope state. The sw-MLPL version is one program that runs the
  same way from REPL, compiled-app, or `mlpl-repl --connect`.
- Patching, embedding, attention, and the loss are all expressible in
  ordinary MLPL operators (matmul, transpose, reshape, softmax) once
  a small number of new builtins land -- no PyTorch-shaped hidden
  state, no `nn.Module` indirection.
- The attention map over patches is a natural use of `svg(...,
  "heatmap")` already shipped in Saga 7.
- Wrapping the train block in `device("mlx") { ... }` routes the
  heavy work to the Apple Silicon peer (Saga R1, v0.18.0) without
  changing the source; the same program runs on CPU, on the in-
  process MLX feature, or against `mlpl-mlx-serve`.
- Typed traces (Saga 23) already cover the producers used here
  (softmax, cross_entropy, attention_weights, grad, weight/bias of
  linear/embed), so the demo gets typed tags and tutoring errors for
  free.

## Repository strategy

Keep this entirely inside `sw-mlpl`. Unlike HRM/BDH/TRM, there is no
external "sw track" repo for ViT yet -- the upstream
`weagan/Vision-Transformer-ViT` is a paper-companion Colab, not a
project we contribute back to. Living in-tree also lets the demos
ship in the web REPL tutorial and the bundled demo dropdown.

Suggested files:

- `demos/vit_attention_pattern.mlpl` (no training; pure viz)
- `demos/vit_single_head_quick.mlpl` (CPU + WASM friendly)
- `demos/vit_multihead_thorough.mlpl` (MLX-targeted)
- `demos/vit_attention_viz.mlpl` (trained-model attention readout)
- A new tutorial lesson "Vision Transformers" in
  `apps/mlpl-web/src/tutorial.rs`.
- A new "Pets: cat vs dog" entry in `apps/mlpl-web/src/demos.rs` once
  the data fixture is in place.

## MLPL support needed

The gap analysis from the cat/dog ViT notebook against current v0.19
primitives shows eight missing pieces, grouped by tier.

### Tier 1 -- required for the quick-training demo (single head, RMSNorm, ReLU)

1. **Image data loading.**
   - `load_images(dir, [H, W]) -> [N, C, H, W]` -- decode PNG/JPEG,
     resize, normalize to f64 in `[-1, 1]`. Optional `mean`/`std`
     args for ImageNet-style normalization.
   - `load_preloaded("pets_tiny")` -- a 200-image cat/dog fixture
     embedded in the workspace (small enough to ship in the WASM
     bundle), returns `{X: [N, 3, H, W], Y: [N], names: [str]}`
     similar to the existing tiny shakespeare preload.
   - `fetch_dataset("oxford_iiit_pet")` -- on first use, downloads
     the Oxford-IIIT Pet tarball to `$MLPL_DATA_DIR` (default
     `dirs::data_dir().join("mlpl")`), caches by sha256, returns the
     same `{X, Y, names}` shape. CLI/native only; WASM falls back to
     the preloaded fixture with a one-time warning.
2. **Patch extraction.**
   - `patchify(x, P) -> [B, N, P*P*C]` over a `[B, C, H, W]` input.
     N = (H/P) * (W/P). Backed by reshape + transpose internally; the
     builtin exists so the demo reads cleanly and the tape has a
     single op to lower.
3. **Sequence concatenation.**
   - `concat(a, b, axis)` for prepending the learnable CLS token
     `[1, 1, D]` (broadcast to `[B, 1, D]`) to the patch sequence
     `[B, N, D]` -> `[B, N+1, D]`. Must be differentiable; tape
     lowering required.

That short list is enough to express a single-head ViT today. The
forward+backward path for `attention(d, 1, seed)`, `rms_norm`,
`residual`, `linear`, `cross_entropy`, `adam`, and `train { }` is
already in place (Sagas 11 and 13).

### Tier 2 -- required for the thorough demo (multi-head, LayerNorm, GELU)

4. **`gelu(x)` builtin + `gelu_layer()`.** Tanh approximation is
   fine (matches PyTorch's `approximate="tanh"`). One builtin in
   `crates/mlpl-runtime`, one tape rule in `crates/mlpl-eval/src/
   grad.rs`, one layer in `crates/mlpl-ml`. Trivial; the work is
   the gradcheck parity test on CPU and MLX.
5. **`layer_norm(d, seed)`.** Mean-centered, unit-variance, learned
   affine `gamma`/`beta`. Existing `rms_norm` is parameter-free; ViT
   uses the learned-affine variant. Add as a Model DSL layer with
   the same shape contract as `rms_norm` and tape lowering parallel
   to it.
6. **Multi-head attention on the tape.** `model_tape.rs` currently
   rejects `heads > 1` with "per-head slicing not yet supported."
   Lower the multi-head forward as reshape `[B, T, d] ->
   [B, T, h, d/h]` -> transpose `[B, h, T, d/h]` -> per-head scaled
   dot-product -> recombine. Either hand-write the backward for the
   slice/recombine, or lean on the existing primitive backwards once
   reshape/transpose are on every path. Gradcheck against finite
   differences on `[2, 6, 16]` with `h=2` and `h=4` fixtures.

### Tier 3 -- nice-to-have for browser / MLX ergonomics

7. **u8/f32 dtype on the wire.** `services/mlpl-mlx-serve/src/
   wire.rs` is f64-only today. Image batches are huge in f64; an
   f32 dtype would halve transfer time, u8 raw pixels would
   quarter it for the load-then-normalize-on-peer path. Not a
   blocker -- the orchestrator can keep doing f64 -- but it's the
   first concrete pull on the "broader dtype coverage" follow-up
   from R1.
8. **Web UI re-routing to `mlpl-serve`.** `apps/mlpl-web` still
   runs entirely in WASM and cannot talk to the server today. The
   "train in the browser, MLX does the work" story stays
   hypothetical until this lands. Already deferred from Saga 21;
   ViT is one of the clearest motivators for picking it up.

### Optional / cosmetic

- `attention_map_over_patches(model, image)` helper that returns the
  CLS-token row of the attention matrix reshaped over the patch grid
  for direct `svg(..., "heatmap")` rendering.
- `save_model(model, path)` / `load_model(path)` -- still deferred
  from Saga 15. The thorough demo can train end-to-end without it;
  the attention-viz demo is more interesting with a frozen
  checkpoint.

## Demo shape

Hyperparameters match the upstream notebook for direct comparison:
`IMG_SIZE = 128`, `PATCH_SIZE = 16`, `D_MODEL = 128`, `MLP_DIM =
256`, `BATCH_SIZE = 32`. Single block. Two output classes (cat=0,
dog=1).

### Demo 1 -- attention pattern (no training)

```mlpl
imgs = load_preloaded("pets_tiny")
x    = batch(imgs.X, 1, 0)            # one image, [1, 3, 128, 128]
P    = patchify(x, 16)                # [1, 64, 768]

W_e  = randn(101, [768, 128])
emb  = matmul(P, W_e)                 # [1, 64, 128]

cls  = randn(102, [1, 1, 128])
seq  = concat(cls, emb, 1)            # [1, 65, 128]
pos  = randn(103, [1, 65, 128])
seq  = seq + pos

a    = attention(128, 1, 201)
A    = attention_weights(apply(a, seq))   # [1, 65, 65]
svg(A, "heatmap")
```

Exercises every Tier 1 builtin, no training, runs in the browser.

### Demo 2 -- single-head ViT, quick training (CPU / WASM)

```mlpl
data = load_preloaded("pets_tiny")
sp   = split(data, 0.8, 0)
Xtr  = patchify(sp.train.X, 16); Ytr = sp.train.Y
Xva  = patchify(sp.val.X,   16); Yva = sp.val.Y

D = 128 ; N = 64

# patch embed + CLS prepend + learnable pos
embedder = linear(768, D, 1)
cls_tok  = randn(102, [1, 1, D])
pos      = randn(103, [1, N + 1, D])

block = chain(
  residual(chain(rms_norm(D), attention(D, 1, 201))),
  residual(chain(rms_norm(D), linear(D, 256, 211), relu_layer(),
                              linear(256, D, 212))))
head  = linear(D, 2, 301)

experiment "vit_quick" {
  train 100 {
    seq = concat(cls_tok, apply(embedder, Xtr), 1) + pos
    out = apply(block, seq)
    cls_row = out * eq(iota(N + 1), 0)   # MLPL has no index op
    logits  = apply(head, reduce_add(cls_row, 1))
    adam(cross_entropy(logits, Ytr), [embedder, cls_tok, pos, block, head],
         0.001, 0.9, 0.999, 0.00000001)
    loss_metric = cross_entropy(logits, Ytr)
  }
}

loss_curve(last_losses)
```

CPU-only, ~200 images, ~5-10 minutes interpreted on an M-class
laptop. Trains in the WASM REPL too if `load_preloaded("pets_tiny")`
ships in the bundle.

### Demo 3 -- multi-head ViT, thorough training (MLX peer)

Same shape as Demo 2, with three changes:

1. `data = fetch_dataset("oxford_iiit_pet")` and a balanced
   sub-sample of ~3700 images (min(cats, dogs) * 2).
2. `attention(D, 4, 201)` instead of `(D, 1, ...)`; `rms_norm` swaps
   to `layer_norm(D, ...)`; `relu_layer()` swaps to `gelu_layer()`.
3. The training block is wrapped in `device("mlx") { ... }`.

```sh
# On the Apple Silicon host
services/mlpl-mlx-serve/release --bind 127.0.0.1:6465

# On the orchestrator host
mlpl-serve --peer mlx=http://127.0.0.1:6465

# On the client
mlpl-repl --connect http://127.0.0.1:6464 -f demos/vit_multihead_thorough.mlpl
```

Once the deferred-Saga-21 web rerouting lands, the same program can
run from the browser pointed at the same orchestrator.

### Demo 4 -- trained-model attention readout

Load a saved checkpoint, decode a single held-out test image,
extract the CLS row of the attention map, reshape it back over the
8x8 patch grid, render as a heatmap overlay on the original image.

Gated on `save_model` / `load_model` shipping; until then this lives
inside Demo 2 / 3 as a tail block instead of a standalone program.

### Compiled-app flow

```sh
mlpl build demos/vit_multihead_thorough.mlpl -o target/vit-demo
target/vit-demo --seed 0 --epochs 30 --out artifacts/run.json
```

If the multi-head tape lowering is the only gap, the compiled path
can ship Demo 2 first and Demo 3 once the lowering lands.

## Phases

1. **Tier 1 builtins.** `load_images`, `load_preloaded("pets_tiny")`,
   `patchify`, `concat` (with tape lowering). Land
   `demos/vit_attention_pattern.mlpl` as the smoke test. CPU-only.
2. **Single-head training demo.** `demos/vit_single_head_quick.mlpl`
   trains end-to-end on the preloaded fixture, validation accuracy
   above the 50% balanced-class baseline within 100 train steps.
   Web REPL runs the same source.
3. **Tier 2 builtins.** `gelu`, `layer_norm`, multi-head attention on
   the tape. Gradcheck parity tests against finite differences for
   each, CPU and MLX.
4. **Thorough training demo.** `demos/vit_multihead_thorough.mlpl`
   trains on the full balanced Oxford-IIIT Pet subset on MLX,
   measurable >=2x speedup vs CPU per epoch and within the same
   final accuracy band as the upstream notebook.
5. **Attention readout demo.** `demos/vit_attention_viz.mlpl`
   loads a trained checkpoint (needs `save_model` / `load_model`,
   currently deferred) and renders attention over patches.
6. **Tutorial + web bundle.** New lesson "Vision Transformers" in
   `apps/mlpl-web/src/tutorial.rs`, new entry in
   `apps/mlpl-web/src/demos.rs`, rebuild `pages/` per the live-demo
   discipline, deploy.
7. **Wire dtype + web rerouting follow-ups.** u8/f32 on the
   `mlpl-mlx-serve` wire; web REPL re-routing to `mlpl-serve`. These
   are independently useful (they fold into the deferred Saga 21
   follow-up saga) and unlock running Demo 3 from the browser.

Phases 1-2 can land before the dev-host move to Linux. Phases 3-4
are natural companions to the existing MLX work and want the Apple
Silicon host. Phase 5 is gated on the deferred save/load work
already in the Saga 15 follow-ups. Phases 6-7 ride the existing
pipelines.

## Acceptance tests

- `demos/vit_attention_pattern.mlpl` returns a `[65, 65]` attention
  map whose row sums are within `1e-6` of 1.0 (rows-of-softmax
  invariant) and renders an SVG heatmap.
- `demos/vit_single_head_quick.mlpl` trains on the preloaded fixture
  with a fixed seed and beats a 60% validation-accuracy threshold
  within 100 steps. Determinism: identical loss curve to within
  `1e-9` on rerun for the same seed.
- `demos/vit_multihead_thorough.mlpl` trains for one epoch
  identically (within fp32 tolerance) on CPU and MLX for a fixed
  seed. Final accuracy on the balanced split lands within the band
  the upstream notebook reports for the same hyperparameters and
  epoch count.
- The Tier 2 builtins each have gradcheck parity tests on CPU and
  MLX, matching the Saga 14 step 006 invariant.
- The new tutorial lesson runs end-to-end via the "Run All" path
  introduced in the recent tutorial work.
- The README demo dropdown lists all four ViT demos and they all
  produce output (Demo 4 may print "load_model not available" until
  checkpointing ships).

## Open questions

- **WASM image decode.** The browser bundle has no `image` crate at
  runtime today. Either ship a pre-decoded `Vec<f32>` fixture inside
  the WASM binary (small, deterministic, ~5-10 MB for 200 128x128x3
  images at f32) or compile `image-rs` into the WASM target. The
  preloaded fixture path picks option 1; `fetch_dataset` on native
  picks option 2.
- **Color channel order.** PyTorch / Pillow / `image-rs` disagree on
  channel ordering. Pin to RGB and `[B, C, H, W]` to match the
  upstream notebook; document the convention in the `load_images`
  builtin docstring.
- **Memory footprint on MLX peer.** Full Oxford-IIIT Pet at f64 is
  ~3.5 GB for the balanced subset at 128x128x3. The orchestrator
  should stream batches rather than ship the full tensor; this is
  the first dataset where the existing "send once, train many" peer
  pattern from `tiny_lm_mlx.mlpl` falls over. Either add batch-level
  forwarding or land the f32 wire dtype before Demo 3.
- **Multi-block ViT.** The upstream notebook is one block. A
  stacked-block follow-up demo wants only `chain` of N copies of the
  block -- no new primitives. Worth a separate `vit_deep.mlpl` once
  the single-block path is stable, but explicitly out of scope here.

## References

- Upstream Colab notebooks (cat/dog ViT, basic and balanced
  variants):
  - <https://github.com/weagan/Vision-Transformer-ViT>
- "An Image is Worth 16x16 Words: Transformers for Image Recognition
  at Scale" (Dosovitskiy et al., 2020):
  <https://arxiv.org/abs/2010.11929>
- Oxford-IIIT Pet dataset:
  <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- Related sw-MLPL plans:
  - `docs/HRM-demo-plan.md`
  - `docs/BDH-demo-plan.md`
  - `docs/TRM-demo-plan.md`
  - `docs/SmolLM2-demo-plan.md`
  - `docs/TTT-demo-plan.md`
- Related sw-MLPL infrastructure:
  - `docs/milestone-ml.md`, `docs/milestone-mlx-backend.md`
  - `docs/using-mlx-service.md`, `docs/using-cli-server.md`
  - `docs/missing-demos.md` (Saga 11 audit; multi-head tape
    lowering gap originates here)
