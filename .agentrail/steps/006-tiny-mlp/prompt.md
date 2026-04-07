Add a tiny one-hidden-layer MLP demo on a non-linearly separable dataset.

1. demos/tiny_mlp.mlpl: build a non-linearly separable 2D dataset (two moons or XOR-style blobs - whichever is easier with the existing primitives). Define a 2 -> 8 -> 2 (or 2 -> 8 -> K) MLP with tanh_fn activation and a softmax output. Train with gradient descent. After training: render loss_curve and boundary_2d so the user can see the boundary curve around the data in a way the linear classifier from step 005 cannot.
2. Compare side-by-side in the demo by also training a linear classifier on the same data and showing both confusion matrices.
3. Add the demo to the web dropdown and a tutorial lesson "Going Non-Linear: A Tiny MLP".
4. Rebuild pages.
5. Allowed: demos/, apps/mlpl-web/src/, docs/