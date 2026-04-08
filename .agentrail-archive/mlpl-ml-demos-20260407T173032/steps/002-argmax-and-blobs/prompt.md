Add the dataset and reduction primitives needed by k-means and the classifier demos.

1. argmax(a, axis) built-in: returns the index of the maximum element along a given axis. Output rank is one less than input rank. Add an analogous axis-less form argmax(a) that returns a scalar.
2. blobs(seed, n_per_class, centers) built-in: takes a length-K*2 vector of (cx, cy) center pairs (or a Kx2 matrix), and returns a tuple-of-sorts: in MLPL terms, return an Nx3 matrix where the first two columns are points (gaussian noise around the assigned center, sigma 0.15) and the third column is the integer class label. N = K * n_per_class. Deterministic given seed.
3. TDD: write tests for argmax shape and correctness on a 2D matrix; write tests for blobs (shape, label distribution, determinism, points clustered near their centers).
4. Update docs/lang-reference.md and docs/usage.md.
5. Allowed: crates/mlpl-array (if needed for argmax), crates/mlpl-runtime, docs/