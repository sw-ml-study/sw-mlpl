Add high-level analysis/visualization helper functions that
generate large/complex SVG diagrams from data, analyses of
data, or views over data.

Motivation: svg(data, type) is a low-level primitive. Real
demos need higher-level helpers that compute the right view
of the data and produce a publication-quality diagram.

1. Add new built-ins (each returns Value::Str via the same
   FnCall special-case path used by svg()):

   - hist(data, bins) -> svg
       Compute a histogram from a vector and render as a bar
       chart with bin edges. `bins` is an integer count.

   - scatter_labeled(points, labels) -> svg
       Nx2 points + length-N labels (cluster ids). Each label
       gets a distinct color from a palette.

   - loss_curve(losses) -> svg
       Vector of loss values, rendered as a line plot with
       Y-axis log option, axis ticks, and a "loss" label.

   - confusion_matrix(predicted, actual) -> svg
       Two vectors of class ids. Compute the confusion matrix
       and render as a heatmap with cell value overlays.

   - boundary_2d(grid_outputs, grid_dims, points, labels) -> svg
       Render a 2D classifier decision boundary by treating
       grid_outputs as a heatmap with training points overlaid.

   - layout_grid(svgs) -> svg  (stretch goal)
       Combine multiple svg strings into a single grid layout.
       Useful for "compare runs" panels.

2. New module: crates/mlpl-viz/src/analysis/ with one file per
   helper. Each helper takes &DenseArray (and possibly other
   args) and returns Result<String, VizError>.

3. mlpl-eval: dispatch each new helper name in eval_expr the
   same way svg() is dispatched. Consider extracting a
   "value-returning builtin" registry to avoid hand-rolling
   each one.

4. Tests:
   - Unit tests in mlpl-viz for each helper
   - End-to-end wasm tests showing they return strings starting
     with <svg

5. Docs: update docs/lang-reference.md with the new helpers,
   docs/usage.md with example usage, and add a tutorial lesson
   for "Visualizing Analyses".

6. Add a demo (demos/analysis_demo.mlpl) that walks through
   training a model, computing a confusion matrix, and rendering
   loss curves + boundary in one go.

7. Quality gates pass, pages rebuilt.

Allowed: crates/mlpl-viz, crates/mlpl-eval, demos/, docs/,
apps/mlpl-web, apps/mlpl-repl