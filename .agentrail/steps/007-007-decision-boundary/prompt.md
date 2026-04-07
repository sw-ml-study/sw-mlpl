Add the "decision_boundary" diagram type and a classifier demo.

1. Add helper function to generate a 2D grid of evaluation points
   - grid(x_min, x_max, y_min, y_max, n) -> Nx2 matrix
   - Wire as built-in: grid(bounds, n) where bounds is [xmin,xmax,ymin,ymax]

2. render_decision_boundary in mlpl-viz:
   - Inputs: grid points (Nx2), classifier outputs at each grid point
     (vector of length N), and original training data with labels
   - Render heatmap of classifier outputs over the 2D space
   - Overlay scatter of training points colored by true label
   - Returns full SVG

3. Wire as svg(grid_outputs, "decision_boundary", training_data)
   - The 3-arg variant of svg() (optional opts is a third positional)

4. Demo: train softmax/logistic on 2D toy dataset (XOR or two
   moons via simple geometry), then render decision boundary

5. Add demo to web demo selector

Note: this step depends on softmax/argmax being available. If
they aren't built-in yet, scope this step to logistic regression
on a linearly separable dataset (e.g., AND gate extended to a
2D point cloud) and defer multi-class to later.

Allowed: crates/mlpl-viz, crates/mlpl-runtime, crates/mlpl-eval,
demos/, apps/mlpl-web/src/