Add "line" and "bar" diagram types to svg().

1. render_line(data, opts) -> String
   - Vector input: x = 0..N-1, y = element values (loss curves)
   - Nx2 matrix input: (x, y) points connected by polyline
   - Auto-scale axes
   - Smooth polyline, with optional dot markers

2. render_bar(data, opts) -> String
   - Vector input: one bar per element
   - Auto-scale Y axis
   - Equal-width bars with small gaps

3. Wire into svg() dispatch:
   - "line" -> render_line
   - "bar" -> render_bar

4. Tests:
   - svg([0.5, 0.4, 0.3, 0.2, 0.1], "line") -- loss curve
   - svg([3, 1, 4, 1, 5, 9, 2, 6], "bar") -- histogram
   - Both render valid SVG strings
   - Both auto-scale correctly

Allowed: crates/mlpl-viz, crates/mlpl-runtime, crates/mlpl-eval