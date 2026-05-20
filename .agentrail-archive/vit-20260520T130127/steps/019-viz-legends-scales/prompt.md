Saga 29 step inserted: add legends and axis scales to every viz type that currently lacks them. User audit identified 5 gaps:

1. heatmap_grid (mlpl-viz/src/svg/heatmap_grid.rs): each per-head cell has its own min/max colormap but no on-screen indicator of what the scale is. Add a compact 'min/max' label pair below each cell (or a tiny colorbar strip on the right of each cell). Per-cell is required because each head's distribution differs.

2. scatter (mlpl-viz/src/svg/scatter.rs render_scatter): no X/Y axes, no tick labels. Add a minimal axis gizmo + 4 tick labels (xmin, xmax, ymin, ymax) near the corners. Match the style of the heatmap legend's monospace 11pt #cdd6f4 fill.

3. line (charts.rs render_line): no X/Y scale labels. Same minimal approach -- (xmin, xmax) along the bottom and (ymin, ymax) along the left edge.

4. bar (charts.rs render_bar): no Y-axis scale. Show ymin (bottom) / ymax (top) along the left edge of the plot area. X-axis ticks (bar index labels) are noisy for >10 bars so leave those off; the index is implicit.

5. decision_boundary (decision_boundary.rs): no colorbar legend. Reuse the heatmap-style vertical colorbar on the right with the 0..1 (or grid's actual min/max) labels.

Tests: add at least one structural test per change verifying the new SVG contains the expected text label (e.g. assert svg.contains(format!('{:.2}', xmin)) after rendering scatter with known data). Keep existing tests passing.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist (157). Pages rebuild + push so live demos pick up the labels.

Stylistic note: keep the additions compact (small font, edge-of-plot placement). The viz is the data; labels exist so the user can read off magnitudes without clicking through the SVG download.