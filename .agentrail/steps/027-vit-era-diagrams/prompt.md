Saga 29 step inserted: SVG diagrams for ViT-era concepts that don't have visual companions yet.

The diagrams/ directory ships SVG explainers for older concepts (the Docs tab and learning paths' Step::Diagram render them). The ViT-era additions don't have diagrams yet. Add:

1. patchify.svg: a 64x64 image split into a 4x4 grid of 16-pixel patches; arrows show the flattening into [16, 768] token vectors.

2. multi-head-attention.svg: input -> Q/K/V projections -> reshape to per-head subspaces -> per-head softmax(QK^T)V -> stack along column axis -> output projection. Annotate one head with the [T, d/h] shape labels.

3. stack-tape-op.svg: side-by-side comparison. Left: O(N^2) chained binary concat. Right: single Stack node with N parents. Highlight the depth difference.

4. vit-pipeline.svg: full ViT forward path on a pet photo. patchify -> linear embed -> +CLS -> +pos -> attention -> take(_, 0) for CLS pooling -> MLP classifier.

5. result-type.svg: Value::Result branches. Ok(payload) and Err(payload). Show the accessor surface (is_ok, unwrap, unwrap_or, err_message).

6. heatmap-grid-viz.svg: [N, R, C] tensor unfolding into a 2x2 grid of N=4 colored heatmaps, each with its own min/max strip.

7. upload-result-flow.svg: file picker -> Canvas decode -> Result. Branches: success -> Ok({pixels, h, w}); dismiss -> Err("cancelled"); bad file -> Err("decode failed: not a valid image").

Implementation: each diagram is a small SVG (300-500 px wide), drawn by hand in the file. Add entries to apps/mlpl-web/src/diagrams.rs so the Diagrams browser surfaces them, plus add Step::Diagram entries to the Vision learning path and other relevant paths.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist; pages rebuild + push.