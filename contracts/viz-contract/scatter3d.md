# `scatter3d` Viz Contract (Saga 16 step 003)

## Purpose

`svg(pts, "scatter3d")` renders a rank-2 array as a
static 3-D scatter plot via orthographic projection.
Complements the 2-D `svg(pts, "scatter")` for
embedding-visualization workflows where a 2-D projection
loses too much structure.

## Input shapes

- **`[N, 3]`** -- pure point cloud. All dots rendered
  with the neutral default color (`#89b4fa`).
- **`[N, 4]`** -- points + integer cluster id in the
  last column. Each cluster gets its own color from the
  standard `analysis::scatter_labeled` palette, cycled
  modulo the palette length (8 colors). A legend swatch
  list appears at the top-right of the canvas.

Any other rank or last-dim is rejected with
`VizError::InvalidShape`.

## Camera

Orthographic projection, fixed azimuth and elevation:

- `azimuth = 30 degrees` (rotation about the world Y
  axis)
- `elevation = 20 degrees` (tilt above the horizontal)

Camera math:

```
x_2d =  x * cos(az) - y * sin(az)
y_2d = (x * sin(az) + y * cos(az)) * sin(el) + z * cos(el)
```

The projection is deterministic and identical across
every call, so snapshot tests of the rendered SVG are
stable.

## Rendering

1. Write the shared SVG header (`write_svg_open` from
   `svg/mod.rs`: dark background, frame lines).
2. Render axis gizmos at a fixed canvas position (bottom-
   left corner, inside the pad): three 24-pixel arrows
   along the projected X/Y/Z axes, each color-coded
   (`#f38ba8` red-ish X, `#a6e3a1` green Y, `#89b4fa`
   blue Z) and labeled with the axis letter at the tip.
   Axes are NOT scaled with the data -- the gizmo
   indicates camera orientation regardless of the point
   cloud's extent.
3. Project every data point, compute the 2-D bounding
   box, and render each as a `<circle>` of radius 3
   using `scale` from `svg/mod.rs` to map data-space
   coordinates to pixel space.
4. If the 4th column is present, render a legend group
   (`<g class="legend">`) at the top-right: a swatch +
   cluster-id text per unique label, sorted ascending.

## Deterministic output

- Points are rendered in row order.
- Legend entries are sorted by cluster id ascending.
- Color assignment is `PALETTE[cluster_id % 8]`.
- All floats in the SVG use the same `{:.1}` format
  specifier as the 2-D renderers.

Snapshot tests on the unit-cube-corners fixture pin
projection + element-order regressions.

## Error cases

All errors surface as `VizError::InvalidShape(msg)`.

- **Non-rank-2 input.** "scatter3d expects rank-2 [N,
  3] or [N, 4], got shape ...".
- **Wrong last-dim.** "scatter3d expects 3 or 4
  columns, got N".
- **Empty input** (`[0, 3]` or `[0, 4]`). Renders an
  empty plot (header + axes + no dots + no legend);
  does not error.

## What this contract does NOT cover

- **Rotation / interactive 3-D.** Scope cut for Saga
  16. A follow-up saga could swap in a WebGL or
  SVG-based rotator without changing the input surface.
- **Perspective projection.** Orthographic only.
- **Depth sorting / occlusion.** Points render in row
  order; closer points may occlude farther ones
  depending on which was submitted later. Not ideal for
  dense clouds but fine for the embedding-viz use case
  (few dozen to few hundred points).
- **Axis ticks and labels at scale.** Only the fixed
  gizmo; no numeric tick marks.
- **Per-point size or alpha encoding.** Uniform radius
  3, no transparency.

## Related

- `crates/mlpl-viz/src/svg/scatter3d.rs` --
  implementation (6 fns, under the 7-fn budget).
- `crates/mlpl-viz/tests/svg_scatter3d_tests.rs` -- 9
  tests covering shape, legend presence, axis gizmos,
  snapshot, and error paths.
- `svg/scatter.rs` -- 2-D sibling renderer.
- `analysis::scatter_labeled` -- the 2-D labeled
  analog; same palette.
