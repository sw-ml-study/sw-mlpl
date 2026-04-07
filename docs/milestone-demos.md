# ML Demos Milestone (v0.4.0)

Building on the v0.3 visualization stack, v0.4 adds the synthetic
data primitives and algorithm glue needed to turn MLPL into a
practical demo platform. Six classic ML algorithms now run end-to-end
in the browser REPL and render publication-quality diagrams inline.

## Delivered

### New built-ins

- [x] `random(seed, shape)` -- seeded uniform `[0, 1)` arrays
- [x] `randn(seed, shape)` -- seeded standard-normal arrays
      (Box-Muller on a private xorshift64 stream, no external crates)
- [x] `argmax(a)` and `argmax(a, axis)` -- flat or per-axis argmax
      (backed by a new `DenseArray::argmax_axis` method)
- [x] `blobs(seed, n_per_class, centers)` -- seeded 2D gaussian-blob
      dataset returning an `Nx3` `[x, y, label]` matrix
- [x] `softmax(a, axis)` -- numerically stable per-axis softmax
- [x] `one_hot(labels, k)` -- `NxK` one-hot encoding

### New demos

- [x] `demos/kmeans.mlpl` -- vectorized Lloyd k-means on a 3-cluster
      blobs dataset, rendered with `scatter_labeled` and the learned
      centers overlaid.
- [x] `demos/pca.mlpl` -- PCA via power iteration on the covariance
      matrix of a linearly-mixed gaussian dataset; renders the
      principal axis as a 2x2 polyline through the centroid.
- [x] `demos/softmax_classifier.mlpl` -- linear softmax + cross-entropy
      classifier on a separable 3-class blobs dataset; renders a
      confusion matrix and a class-0 decision-boundary surface.
- [x] `demos/tiny_mlp.mlpl` -- a 2 -> 8 -> 2 MLP with tanh activation
      and manual backprop on an XOR-style dataset; learns a curved
      boundary the linear model can't, with a loss curve captured via
      an `iota`/`eq` mask trick (MLPL has no array indexing).
- [x] `demos/attention.mlpl` -- scaled dot-product attention pattern
      (`softmax(Q K^T / sqrt(d), 1)`) rendered as heatmaps, including
      a self-attention variant where the diagonal dominates.

### Tutorial and web UI

- [x] New tutorial lessons: "Unsupervised: K-Means",
      "Dimensionality Reduction: PCA", "Multi-class Classification",
      "Going Non-Linear: A Tiny MLP", "Attention Patterns".
- [x] Demo dropdown entries (alphabetized): Attention Pattern,
      K-Means, PCA, Softmax Classifier, Tiny MLP.
- [x] REPL banners bumped from v0.3 to v0.4.

### Tests

- Runtime: `softmax_onehot_tests.rs`, `random_builtins_tests.rs`,
  `argmax_blobs_tests.rs`.
- Eval integration: `kmeans_demo_tests.rs`, `pca_demo_tests.rs`,
  `softmax_classifier_tests.rs` (>95% on blobs), `tiny_mlp_tests.rs`
  (MLP beats linear on XOR), `attention_demo_tests.rs`
  (row sums == 1, diagonal dominates when K = Q).

## Success criteria

- [x] `random`, `randn`, `argmax`, `blobs` built-ins with tests
- [x] K-means demo converges and renders cleanly inline
- [x] PCA demo shows clear separation along the principal axis
- [x] 3-class softmax classifier > 95% accuracy on separable blobs
- [x] Tiny MLP beats linear classifier on non-linearly separable data
- [x] Attention pattern renders as a recognizable heatmap
- [x] All tests pass, all quality gates green, pages deployed
- [x] Tutorial includes at least one lesson per Phase 2 demo

## What's next

- Faster convergence via Adam / momentum optimizers
- Richer datasets (moons, circles, spirals) beyond blobs
- Live training animations using the existing trace pipeline
