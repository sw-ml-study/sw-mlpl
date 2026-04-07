Add "heatmap" diagram type for visualizing matrices.

1. render_heatmap(data, opts) -> String
   - Input: MxN matrix
   - Each cell becomes a colored rectangle
   - Color scale: viridis-like (dark purple -> teal -> yellow)
   - Auto-scale color range to min/max of data
   - Optional cell values overlaid for small matrices
   - Optional row/column labels

2. Color helper functions
   - value_to_rgb(v: f64, min: f64, max: f64) -> (u8, u8, u8)
   - Stay simple: linear interpolation across 3 control points
     for a viridis-like ramp

3. Wire into svg() dispatch: "heatmap" -> render_heatmap

4. Tests:
   - svg(reshape(iota(12), [3, 4]), "heatmap") returns valid SVG
   - svg with all-equal values doesn't NaN
   - 1x1 matrix works
   - Large matrix (100x100) renders without choking

Allowed: crates/mlpl-viz, crates/mlpl-runtime, crates/mlpl-eval