Phase 4: Demo integration.

Wire 2-3 demos to emit Stage3dEvents: Basics (arithmetic, arrays, reshape -- scalars flowing into vectors into matrices), Loss Curve (training iterations -- loss value as vertical bar tracking loss height), Moons MLP (forward pass layers -- input -> weights -> output per layer).

Each demo's eval steps automatically emit events when show_3d is true. The 3D stage shows the computation history as a chronological storyboard. Pages rebuild required.