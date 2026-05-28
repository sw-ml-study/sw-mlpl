# components/viz/ migration + decompose (saga 58)

Move mlpl-viz (13 modules, FAIL) into components/viz/ and split
into sparse sibling crates. Most viz logic is free functions, so
the split is mechanical (no extension-trait pattern needed).

## Sibling plan

- mlpl-viz-error: VizError type (shared)
- mlpl-viz-svg-core: SVG base rendering
- mlpl-viz-svg-scatter
- mlpl-viz-svg-heatmap (heatmap + heatmap_grid)
- mlpl-viz-svg-gallery (gallery + gallery_layout)
- mlpl-viz-svg-boundary (decision_boundary + critical_dimensions + boundary_2d_validate)
- mlpl-viz-3d (plotly3d)
- mlpl-viz-analysis
- mlpl-viz: facade re-exporting public API

## Steps

1. scaffold-and-move: create component, move mlpl-viz in.
2. decompose: split into siblings.
3. close.
