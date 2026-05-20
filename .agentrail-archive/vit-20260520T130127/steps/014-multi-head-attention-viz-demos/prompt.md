Saga 29 step inserted between 013 (multi-head tape) and the renumbered thorough demo: ship two demos that visualize single vs multi-head attention. Step 013 made multi-head trainable on the autograd tape; this step makes the architectural payoff visible.

Scope (one PR / one step):

1. demos/vit_attention_pattern_multihead.mlpl -- CPU-only, no-training. Same synthetic image and patchify+linear-embed+CLS+pos pipeline as the existing demos/vit_attention_pattern.mlpl, but attention(128, 4, ...) instead of (128, 1, ...). The result of attention_weights(mdl, X) is [4, 17, 17]. Emit four [17, 17] heatmaps composed into a single 2x2 grid SVG via a small composition helper (each child heatmap wrapped in a <g transform='translate(...)'> so the four sit in quadrants with head labels). Untrained, so heads look uniform-random -- baseline before training.

2. demos/vit_multihead_quick.mlpl -- the multi-head sibling of vit_single_head_quick.mlpl. Same 20-image balanced cat/dog subset, same 100-step adam loop, same MLP classifier head, but attention(128, 4, ...). After training, run attention_weights on one held-out test image and render the four [17, 17] heatmaps in a 2x2 grid (reuse the helper from demo 1). The point: heads specialize through gradient descent without architectural hand-coding.

3. SVG grid composition helper -- new private helper in mlpl-viz (e.g., svg_grid(svgs, rows, cols, cell_w, cell_h)) that wraps existing svg() outputs into one composed SVG. Used by both demos. Keep it minimal -- the helper exists to express the 2x2 layout, not to be a full layout engine.

4. apps/mlpl-web/src/demos.rs -- add both demos to the DEMOS array with About/intro text explaining the untrained-vs-trained story.

5. apps/mlpl-web/src/lessons.rs -- NOT in scope for this step; tutorial lesson is a follow-up.

6. Tests -- one integration test per demo confirming the demo parses + evals end-to-end and the composed SVG has four <g> children (one per head). The trained variant should be #[ignore]'d like vit_single_head_quick.

Quality gates as usual: cargo test/clippy/fmt/markdown-checker/sw-checklist, README demo count drift (26 -> 28 if both ship), rebuild pages/ via scripts/build-pages.sh, commit + push.

Why this slots in here: the user asked 'what does multi-head attention show/demonstrate' right after step 013 shipped, then asked for visualization demos. This step makes the just-unblocked capability tangible before the heavier step 014 (multihead-thorough-demo on MLX) takes 30+ epochs on a real dataset.