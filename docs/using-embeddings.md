# Using MLPL for Embedding Visualization

> **Status:** reference. Shipped in Saga 16 (v0.14.0);
> Saga 16.5 (v0.14.1) added `pca(X, k)` and
> `embed_table(model)` as convenience builtins on top
> of the same surface. `docs/plan.md` Saga 16 is the
> original design sketch; this doc is the shipped-
> surface reference + honest retrospective.

## What this is about

When you train a model with an `embed` layer (or any
layer that produces a rank-2 `[N, D]` representation),
the immediate next question is "what did it actually
learn?". For that, visualization beats staring at a
numeric matrix.

Saga 16 adds three small builtins plus one new SVG type
that close the loop:

- `pairwise_sqdist(X)` returns the `[N, N]` squared-
  Euclidean distance matrix.
- `knn(X, k)` returns each row's `k` nearest non-self
  neighbors sorted by ascending distance.
- `tsne(X, perplexity, iters, seed)` runs classic van
  der Maaten t-SNE to reduce `[N, D]` to `[N, 2]`.
- `svg(pts, "scatter3d")` renders `[N, 3]` or `[N, 4]`
  (points + cluster id) as an orthographic 3-D
  scatter with axis gizmos and optional legend.

These compose with the existing `embed` / `apply` /
`svg` surface without any new language concepts.

## The builtins

### `pairwise_sqdist(X) -> D`

Shape-preserving pairwise distance. `D[i, j] =
sum_k (X[i, k] - X[j, k])^2`. Symmetric; zero
diagonal; empty input returns empty `[0, 0]`. Contract:
`contracts/eval-contract/pairwise-sqdist.md`.

Used by `knn` internally, but also stands on its own
when you want the full distance matrix (e.g. for a
hierarchical clustering or a custom reduction).

### `knn(X, k) -> idx [N, k]`

Index-returning k-nearest-neighbor query, self-
excluded. For each row `i`, returns the `k` indices of
the closest other rows by squared Euclidean distance,
sorted by ascending distance with ties broken by lower
original index (stable sort). Shape
`[N, k]` of integer-valued `f64` indices.

Self-exclusion is a fixed semantics: `i` never appears
in `knn(X, k)[i]`. That is why `k < N` rather than
`k <= N`; a self-inclusive variant is a follow-up.

Contract: `contracts/eval-contract/knn.md`.

### `tsne(X, perplexity, iters, seed) -> Y [N, 2]`

Classic van der Maaten t-SNE, CPU-only. Constructs a
high-dim joint probability matrix via per-row
perplexity-calibrated binary search; fits a low-dim
Student-t affinity matrix; minimizes KL via momentum-
based gradient descent with early exaggeration.

Hyperparameters are the van der Maaten defaults baked
into the builtin:

- Learning rate 200.
- Early-exaggeration factor 4 for the first 100
  iterations.
- Momentum 0.5 for the first 250 iterations, 0.8
  thereafter.
- Per-row binary search: 50 iters max, entropy
  tolerance 1e-5.
- Probability clamp: 1e-12.
- Initial `Y`: `randn(seed, [N, 2]) * 1e-4`.

Determinism: two calls with identical `(X, perplexity,
iters, seed)` produce bit-identical `Y`. Rotational +
reflection symmetry: the KL loss is invariant under
arbitrary 2-D rotations/reflections of Y, so absolute
coordinates are seed-sensitive. Tests assert
structure-preservation (cluster centroids pairwise-
distinct), not exact coordinates.

Contract: `contracts/eval-contract/tsne.md`.

## The viz type

### `svg(pts, "scatter3d")`

Static 3-D scatter via orthographic projection at a
fixed azimuth=30 / elevation=20 camera. Accepts `[N,
3]` (pure point cloud, neutral color) or `[N, 4]`
(points + integer cluster id, with palette-colored
circles and a sorted legend). Three labeled axis
gizmos at the bottom-left indicate camera orientation;
circle positions are scaled to canvas pixel space from
the projected 2-D bounds.

Deterministic output -- snapshot tests on the unit-
cube-corners fixture pin projection and element-order
regressions.

No rotation, no interactivity. A static snapshot for
docs and "is my cluster structure real?" checks.

Contract: `contracts/viz-contract/scatter3d.md`.

## PCA is shipped as a builtin (v0.14.1)

```
pca(X, k) -> Y
```

- `X` rank-2 `[N, D]`, all finite.
- `k` positive integer, `1 <= k <= D`.
- `Y` rank-2 `[N, k]` -- centered-and-projected data.

Returns the projected data only, not the eigenvectors.
Callers who need the components themselves can still
run the power-iteration + deflation composition
directly (the "Dimensionality Reduction: PCA" tutorial
lesson, Saga 8), which is a useful pedagogical
reference for understanding what the builtin does:

```mlpl
Cov = matmul(transpose(Xc), Xc) / N
v1 = [1, 0, ..., 0]
repeat 50 { v1 = matmul(Cov, v1) ; v1 = v1 / sqrt(dot(v1, v1)) }
# Deflate: Cov_2 = Cov - lambda1 * (v1 @ transpose(v1))
# Repeat for v2, v3...
```

The builtin does the same thing, plus Gram-Schmidt
inside the power-iteration loop so that `V` stays
orthonormal to machine precision even when later
eigenvalues are numerical noise. That detail matters
when `k = D` on rank-deficient or near-rank-deficient
inputs.

Contract: `contracts/eval-contract/pca.md`.

## Extracting embed-layer weights (v0.14.1)

```
embed_table(model) -> table
```

Walks a `ModelSpec` tree depth-first left-to-right and
returns the first Embedding layer's `[vocab, d_model]`
lookup table, cloned from `env`. Closes the Saga 16
gap where a trained
`chain(embed, transformer_block, head)` had no
source-level way to pull the learned embedding
weights back out -- `apply(standalone_embed, iota(V))`
only worked when the embed was a top-level standalone
model.

The natural flow becomes a one-liner:

```mlpl
m = chain(embed(V, d, 0), transformer_block, head)
train N { adam(cross_entropy(apply(m, X), Y), m, ...) }
table = embed_table(m)
emb_2d = tsne(table, 5.0, 300, 0)
svg(emb_2d, "scatter")
```

First-match semantics: if a model somehow contains two
Embedding layers (not a shipped pattern; encoder/
decoder stacks with separate embeddings are out of
scope), `embed_table` returns the first one
encountered. A path-selector variant
(`embed_table(model, "encoder.embed")`) is a
deferred follow-up.

`Residual(Embedding)` is recognized structurally; at
apply time the residual's shape math is ill-typed
(embed changes input shape), but the `embed_table`
walk is purely structural and does not call apply.

Contract: `contracts/eval-contract/embed-table.md`.

## The shipped demo

`demos/embedding_viz.mlpl` walks the pipeline end-to-
end (CPU, any host):

1. Build a structured 3-cluster target in 8-D:
   `target = matmul(cluster_assign, centers) +
    randn(1, [12, 8]) * 0.1` where cluster_assign is
   a `[12, 3]` one-hot matrix and centers is a
   `[3, 8]` matrix with each row concentrating on one
   of the first three coordinate axes.
2. Train a standalone `emb = embed(V, d, 0)` toward
   the target via mean-squared error for 50 adam
   steps.
3. Extract the learned table:
   `table = apply(emb, iota(V))`. The
   `apply(embed, iota(V))` pattern gives you the
   layer's weights; the onehot of the identity token
   sequence is the identity matrix, so
   `onehot @ table = table`.
4. t-SNE to 2-D: `tsne(table, 3.0, 300, 7)`; render
   via `svg(..., "scatter")`.
5. 3-D via `pca(table, 3)` (v0.14.1); render via
   `svg(..., "scatter3d")`.
6. `knn(table, 3)` reports each token's 3 nearest
   non-self neighbors. After training, every token
   in cluster 0 (ids 0-3) has neighbors from cluster
   0; similarly for clusters 1 (4-7) and 2 (8-11).

Run:

```bash
./target/release/mlpl-repl -f demos/embedding_viz.mlpl
```

Final tail of the `neighbors` output (last 5 rows
i=7..11): `4 5 6` | `11 9 10` | `11 10 8` | `11 9 8`
| `9 10 8`. Every listed index belongs to the token's
own cluster -- the learned embedding recovered the
target structure.

### The web REPL lesson

The "Embedding exploration" lesson at
<https://sw-ml-study.github.io/sw-mlpl/> runs a 6-
point fixture in 3-D (two clusters of three) -- small
enough that every t-SNE call returns in milliseconds
in WASM. Covers the same four operations
(`pairwise_sqdist`, `knn`, `tsne`, `svg(...,
"scatter3d")`) at a size where you can eyeball every
output.

## Training inside a chain (v0.14.1)

The natural demo shape for "learn embeddings, then
visualize" is:

```mlpl
model = chain(embed(V, d, 0), transformer_block, head)
train N { adam(cross_entropy(apply(model, X), Y), model, ...) }
table = embed_table(model)
```

v0.14.1 ships the last line. Before v0.14.1,
`chain(...)` evaluation resolved bare identifiers only
via `env.vars` (not `env.models`), so a pre-bound
`emb_layer = embed(V, d, 0)` binding could not be
referenced inside a `chain(...)` call -- and there was
no source-level way to reach "the embed sublayer of
this chain". The shipped `embed_table(model)` builtin
walks the `ModelSpec` depth-first, first match wins,
and returns the Embedding layer's `[vocab, d_model]`
table cloned from `env`.

The standalone-embed + MSE-to-target approach used in
`demos/embedding_viz.mlpl` still works and is a
smaller pedagogical starting point. The training-
inside-a-chain flow is the path once you're running a
real model end-to-end.

## Parity testing

Five test files pin the surface:

- `crates/mlpl-eval/tests/pairwise_sqdist_knn_tests.rs`
  (13 tests) -- 3-point / 4-point fixtures for
  `pairwise_sqdist` (symmetry, zero diagonal, empty
  input) and `knn` (nearest-on-a-line, tie-break,
  self-exclusion, error paths).
- `crates/mlpl-eval/tests/tsne_tests.rs` (9 tests) --
  structure-preservation on 3 Gaussian blobs in [60,
  4], shape preservation, determinism under identical
  seeds, seed-sensitivity, every error path.
- `crates/mlpl-viz/tests/svg_scatter3d_tests.rs`
  (9 tests) -- shape validation, `<circle>` count,
  legend presence, axis labels, unit-cube-corners
  snapshot, error paths, empty input.
- `crates/mlpl-eval/tests/embedding_viz_tests.rs`
  (4 tests) -- cut-down V=9/d=4 end-to-end pipeline
  (tsne shape, knn invariants, 3-D projection shape,
  svg renders produce well-formed SVG).

## Not shipped (deferred follow-ups)

- **UMAP.** Sibling nonlinear reducer to t-SNE.
  Overlaps in user experience for marginal value;
  doubling the testing surface was out of Saga 16's
  scope. Ships once a concrete use case justifies the
  second algorithm.
- **RAG pipeline.** Real RAG wants a local LLM
  inference path for the "retrieve + augment + LLM
  generation" story; without that the demo is
  performative. Follow-up saga after Saga 19 (LLM-as-
  tool REST).
- **Interactive 3-D scatter (rotation, zoom).** The
  shipped `scatter3d` is a static snapshot. A WebGL
  or SVG-based rotator is a natural follow-up without
  changing the input surface.
- **MLX dispatch for `tsne`.** Its inner loop (per-
  point perplexity binary search, SGD over pairwise
  affinities) does not vectorize cleanly through
  MLX's kernel model at embedding-table scale. CPU-
  only today. `pairwise_sqdist` and `knn` are CPU-
  only primitives too; both are small and dominated
  by launch overhead at our scales.
- **Approximate / Barnes-Hut t-SNE.** Exact
  `O(N^2)` is fine at embedding-table scale;
  approximate methods land once anyone runs t-SNE on
  `N >= 10000`.

## Related

- `contracts/eval-contract/pairwise-sqdist.md` --
  distance matrix contract.
- `contracts/eval-contract/knn.md` -- k-NN contract.
- `contracts/eval-contract/tsne.md` -- t-SNE
  hyperparameters + algorithm.
- `contracts/eval-contract/pca.md` -- top-k PCA
  builtin (v0.14.1).
- `contracts/eval-contract/embed-table.md` --
  embed-table extraction builtin (v0.14.1).
- `contracts/viz-contract/scatter3d.md` -- 3-D
  scatter camera + rendering.
- `demos/embedding_viz.mlpl` -- CPU demo.
- `docs/demos-scripts.md` Demo 9 -- run guide.
- Saga 8 "Dimensionality Reduction: PCA" tutorial
  lesson -- PCA composition pattern.
- Sibling docs: `docs/using-perturbation.md`
  (Saga 20), `docs/using-lora.md` (Saga 15).
