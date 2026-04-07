Implement the svg() built-in with the "scatter" diagram type.

1. Create crates/mlpl-viz/src with an svg module
   - fn render_scatter(data: &DenseArray, opts: SvgOpts) -> String
   - Input: Nx2 matrix of (x, y) points
   - Output: complete <svg>...</svg> string
   - Auto-scale axes to data range
   - White background, dark dots, light grid
   - 400x300 default size

2. Wire svg() built-in in mlpl-runtime (or mlpl-eval since
   it returns a Value::Str now):
   - svg(data, type_name) where type_name is a Str arg
   - Match on type_name: "scatter" -> render_scatter(...)
   - Other types return error: "unknown svg type"
   - Returns Value::Str(svg_string)

3. Tests:
   - svg([[0,0],[1,1],[2,4]], "scatter") returns string starting "<svg"
   - String contains expected SVG element count
   - Empty data renders an empty plot
   - Wrong shape (vector instead of matrix) returns clean error
   - Unknown type returns clean error

4. Keep mlpl-viz placeholder doc removed; make this its first
   real feature.

Allowed: crates/mlpl-viz, crates/mlpl-runtime, crates/mlpl-eval