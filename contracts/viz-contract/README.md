# Viz Contract

## Purpose

Define the behavioral spec for trace visualization in MLPL. `mlpl-viz`
renders trace data into visual formats. This is post-MVP scope -- MVP
uses CLI + JSON trace export only.

## Key Types and Concepts

### Renderer

Takes a `Trace` and produces visual output.

- Timeline view: step-by-step execution replay
- Array view: 2-D heatmap or table for array values
- Future: Yew/WASM interactive viewer

### svg() dispatch (current)

`svg(data, "type"[, aux])` dispatches by `type` string to one of
the `render_*` functions. Supported types:

- `"line"`, `"bar"` -- rank-1 vector charts.
- `"scatter"`, `"scatter3d"` -- rank-2 point matrices.
- `"heatmap"` -- rank-2 matrix as a viridis grid.
- `"decision_boundary"` -- rank-2 grid + rank-2 training points
  via `aux`.
- `"gallery"` -- Saga 29 step 010: rank-4 `[N, 3, H, W]` image
  batch rendered as an SVG grid of RGB thumbnails. Values
  expected in `[-1, 1]`-normalized space (the same range
  `load_preloaded("pets_tiny")` and `load_images` emit);
  out-of-range values clamp instead of wrap. Thumbnails are
  downsampled via block averaging so a `[20, 3, 64, 64]`
  pets_tiny slice renders in ~5 MB of SVG rather than 80K
  unique `<rect>` elements.

## Invariants

- Rendering must not modify the trace
- Visual output must faithfully represent trace data

## What This Contract Does NOT Cover

- Trace recording (that is `mlpl-trace`)
- Evaluation logic
- CLI output formatting (that is in apps)
- WASM/Yew build infrastructure (post-MVP)

## Open Questions

- Output formats: SVG? HTML? Terminal-based?
- Whether viz is a library or an app
- When the Yew/WASM viewer enters scope
