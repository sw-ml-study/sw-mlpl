Phase 2a step 3: Connection arrows between steps.

When a step's label references a variable defined in a previous step (e.g., step 3 'y = matmul(W, x)' references step 1 'x' and step 2 'W'), draw a thin curved line from the source sculpture to the consuming sculpture. Parse labels for variable references, match against previously emitted step names.

Render as THREE.TubeGeometry following a quadratic bezier curve lifted slightly above the ground. Color: dim gray. Pages rebuild required.