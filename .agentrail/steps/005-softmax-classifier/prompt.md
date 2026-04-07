Add a 3-class softmax classifier and confusion-matrix demo.

1. New built-ins (in mlpl-runtime, with TDD): softmax(a, axis) and one_hot(labels, k). softmax stabilizes by subtracting the row max before exponentiation.
2. demos/softmax_classifier.mlpl: generate a 3-class blobs dataset, train a linear softmax classifier with gradient descent for ~300 steps. After training: render the loss curve, the confusion matrix from rounded predictions vs labels, and the per-class boundary on a 2D grid using boundary_2d (use the dominant predicted class as the grid output, and the original blobs points as the overlay).
3. Add the demo to the web dropdown and a tutorial lesson "Multi-class Classification".
4. Verify the model reaches > 95 percent accuracy on the (separable) blobs dataset.
5. Rebuild pages.
6. Allowed: crates/mlpl-runtime, demos/, apps/mlpl-web/src/, docs/