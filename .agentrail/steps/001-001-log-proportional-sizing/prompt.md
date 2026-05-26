Phase 2a step 1: Log-proportional sculpture sizing.

Replace linear-proportional mesh dimensions with log2-based sizing in stage3d.js shapeMesh(). Vector width = log2(N) clamped to [0.3, 6]. Matrix: log2(M) x log2(N). Tensor: log2(B) layers. Minimum 0.3 units for scalars, max 6 units.

This makes a [3] vector and a [300] vector visually distinguishable without the [300] dominating the stage. Pages rebuild required.