Saga 29 step 010 (inserted prereq, pulled forward from
Phase 4.5 in docs/milestone-vit.md): gallery-viz.

Why: the trained ViT demo (step 009) outputs only a loss
curve + a scalar accuracy -- there is no way to SEE which
images the model is classifying. Users have asked
"what are these pets and how is the model doing on each?"
and the answer is "we don't show that yet". This prereq
ships an `svg(images, "gallery")` viz output that renders
an `[N, 3, H, W]` tensor as an SVG grid of thumbnails.
Step 010b layers a `predict_batch(model, X)` builtin and a
demo that labels each thumbnail with `actual: cat / pred:
dog` style overlays.

Scope (one PR / one step):

1. mlpl-viz: new "gallery" render path.
   - Signature mirrors the existing svg dispatch: accept a
     rank-4 `[N, 3, H, W]` tensor in `[-1, 1]` (the
     pets_tiny / load_images normalization) or `[0, 255]`
     (raw u8-equivalent); render into an SVG grid.
   - Grid layout: ceil(sqrt(N)) columns, ceil(N / cols)
     rows. Each thumbnail is W x H pixels at the source
     resolution; CSS `image-rendering: pixelated` so they
     don't blur in the browser.
   - Optional auxiliary input: a `[N]` or `[N, 2]`
     overlay tensor that carries predictions/labels.
     Initially out of scope -- the optional overlay lands
     in step 010b's predict_batch wrapper.
   - Implementation: render each pixel as an `<rect>` or
     stream a base64-encoded PNG per thumbnail. PNG approach
     is much smaller bytewise; reuse the `image-io` feature's
     `png::Encoder` for the encoding. Native-only via
     `image-io`; WASM falls back to rect-per-pixel (slower
     to render but no decoder needed in WASM).

2. mlpl-eval / mlpl-viz dispatch in `eval_svg`:
   - Add `"gallery"` as a recognized type name in the svg()
     dispatch alongside `"heatmap"`, `"line"`, etc.

3. Tests (`crates/mlpl-viz/tests/gallery_render_tests.rs`):
   - Render a hand-rolled `[1, 3, 4, 4]` rainbow gradient ->
     SVG bytes -> assert it contains an `<svg` tag,
     viewBox, expected width/height.
   - `[4, 3, 8, 8]` random tensor -> 2x2 grid layout
     verified (svg contains 4 `<image>` / `<rect>` blocks).
   - `[N, 3, 64, 64]` pets_tiny slice (via take + concat
     of the existing fixture): render and assert non-empty
     + correct dimensions.

4. Contracts (`contracts/viz-contract` README): add a
   "gallery" subsection. docs/glossary.md: new "Gallery
   viz" entry. docs/lang-reference.md: row for the new
   svg type.

Quality gates: cargo test (workspace), clippy -D warnings,
fmt, markdown-checker (touched docs), sw-checklist held or
lowered. /mw-cp checkpoint. Commit + push before agentrail
complete.

Followups (out of scope):
- predict_batch + per-thumbnail overlay -- ships in step
  010b right after.
- Attention overlay (mapping attention weights back to
  pixel space) -- separate followup step.
- Animated grid transitions / interactivity -- nice-to-
  have, defer.

Why this insert (versus shipping multi-head first): the
existing demos are correct but illegible without a way to
SEE the images. Step 010b + 010c layer onto this. The
multi-head + thorough demo (now step 012+) is more
impressive once the gallery viz makes the quick demo's
output interpretable.
