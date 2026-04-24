Phase 2 step 003: 3D scatter viz type
`svg(pts, "scatter3d")`.

New viz type in `crates/mlpl-viz` for rendering a 3-D
point cloud as a static SVG via orthographic projection.

1. **Input shapes accepted**:
   - `[N, 3]`: pure point cloud, single default color.
   - `[N, 4]`: points + cluster id in the last column
     (mirrors `scatter_labeled`). Cluster ids are
     integer-valued f64; distinct ids get distinct
     colors from the standard palette
     (`mlpl-viz::palette`).

2. **Projection** (document in the contract):
   - Orthographic at azimuth=30 degrees,
     elevation=20 degrees.
   - `x_2d = x*cos(az) - y*sin(az)`
   - `y_2d = (x*sin(az) + y*cos(az))*sin(el) + z*cos(el)`
   - Rescale the projected 2-D coordinates to a fixed
     canvas size.

3. **Rendering**:
   - Axis gizmos (three short arrows labeled X/Y/Z at
     the origin) rendered first so they sit behind the
     points.
   - Points rendered as `<circle>` elements with the
     per-cluster color when a 4th column is present, or
     a neutral default otherwise.
   - Legend (cluster id -> color) when a 4th column is
     present.
   - Deterministic element order so snapshot tests
     catch projection-math regressions.

4. **Where**: extend `crates/mlpl-viz/src/scatter.rs`
   (or add a sibling `scatter3d.rs` if the file is at
   the sw-checklist budget) with the new render
   function. Wire into the `svg(data, type)` dispatch
   in `crates/mlpl-viz/src/lib.rs` (or wherever the
   type -> renderer map lives).

5. Contract `contracts/viz-contract/scatter3d.md`:
   input shapes, projection formula, azimuth + elevation
   defaults, non-goals (no rotation, no interactive 3-D;
   those are a follow-up saga).

6. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-viz/tests/scatter3d_tests.rs`:
   - Shape validation: rank != 2, last-dim != 3 or 4
     -> error.
   - Rendered SVG string contains N `<circle>`
     elements (the dots).
   - When a 4th column is present, legend appears and
     each unique cluster id gets its own color.
   - Snapshot test against a small fixed fixture (e.g.
     the 8 corners of a unit cube) so regressions in
     the projection math surface immediately.

7. Wire the new type into `mlpl-eval` so
   `svg(pts, "scatter3d")` works from the MLPL source
   level. If the existing dispatch in `eval_ops.rs` or
   `mlpl-viz` already routes by name, just add the new
   arm; no evaluator change otherwise.

8. Quality gates + `/mw-cp`. Commit message references
   Saga 16 step 003.
