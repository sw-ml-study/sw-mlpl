Saga 29 step 011 (inserted prereq, pulled forward from
Phase 4.5 in docs/milestone-vit.md):
predict-batch-and-gallery-demo.

Why: step 010a's `svg(images, "gallery")` renders thumbnails
but with no labels or predictions overlaid -- it's just a
contact sheet. This step adds `predict_batch(model, X)` so a
trained classifier produces per-image labels, plus a new
demo (`demos/vit_predict_gallery.mlpl`) that renders the
pets_tiny slice annotated with `actual: cat / pred: dog`
overlays. After this step the user can finally SEE both the
inputs and the model's answers.

Scope (one PR / one step):

1. mlpl-runtime / mlpl-eval: `predict_batch(model, X) -> Y`
   builtin.
   - For a `[N, ...]` input X, run forward through the
     model and return `[N]` integer predictions (argmax over
     the last axis of the logits output).
   - Works with the rank-3 attention path from step 008 so
     the full ViT classifier round-trips on a real batch.
   - Native + WASM (no image-io requirement -- this is just
     model forward + argmax).

2. mlpl-viz / mlpl-eval: extend `svg(images, "gallery")`
   (step 010a) to accept an optional overlay vector. The
   3-arg form `svg(images, "gallery", overlay)` where
   `overlay` is a `[N]` integer vector or `[N, 2]` matrix
   (cols = actual/predicted) renders each thumbnail with a
   text overlay underneath.

3. demos/vit_predict_gallery.mlpl:
   - Load pets_tiny, build a balanced 16-image subset (8
     cats + 8 dogs), train a tiny ViT for 30 steps (same
     architecture as step 009's web demo).
   - `predict_batch(model, X)` -> [16] predictions.
   - `svg(X, "gallery", concat(Y, preds, 1))` -> labeled
     thumbnail grid.
   - Misclassifications stand out because the actual /
     predicted overlays disagree.

4. Web wiring:
   - apps/mlpl-web/src/demos.rs: new "Pets: predict gallery"
     entry mirroring the demo file (compact narration,
     short training, full predict + gallery render).
   - Update DEMOS count drift + README.

5. Tests:
   - mlpl-eval/tests: predict_batch produces argmax(logits,
     last) on a fixture model + input.
   - mlpl-viz/tests: gallery render with overlay tensor
     produces SVG containing the overlay strings.
   - mlpl-eval/tests: vit_predict_gallery.mlpl end-to-end
     gated #[ignore]'d (heavy training).

6. Contracts + docs: update viz-contract, glossary,
   lang-reference for the 3-arg svg + predict_batch.

Quality gates: cargo test (workspace), clippy -D warnings,
fmt, markdown-checker, sw-checklist held or lowered.
/mw-cp checkpoint. Commit + push before agentrail complete.

Out of scope (followups, not this step):
- BYO-image upload (step 010c in the original plan -- now
  shifts to step 012 after this insert).
- Attention overlay on top of the thumbnails.
- Multi-class display (the demo is binary classification;
  the renderer should still work for K-class but no need
  to demo that yet).

Why this is one step, not folded into 010a or the demo:
gallery-viz (010a) is renderer plumbing; this step adds the
overlay piece + a real classifier + a real demo. Splitting
keeps each step's review surface focused. The two together
finally let the user SEE what the trained ViT is doing on
real pet images.
