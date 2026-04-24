# Using MLPL for Embedding Visualization

> **Status:** reference. Shipped in Saga 16 (v0.14.0).
> `docs/plan.md` Saga 16 is the original design sketch;
> this doc is the shipped-surface reference + honest
> retrospective.

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

## PCA is a composition pattern, not a builtin

MLPL already lets you compute top-k principal
components of a matrix via power iteration and
deflation; see the "Dimensionality Reduction: PCA"
tutorial lesson (shipped in Saga 8). For a rank-2
matrix `Xc` (centered), the loop is roughly:

```mlpl
Cov = matmul(transpose(Xc), Xc) / N
v1 = [1, 0, ..., 0]
repeat 10 { v1 = matmul(Cov, v1) ; v1 = v1 / sqrt(dot(v1, v1)) }
# Deflate: Cov_2 = Cov - lambda1 * (v1 @ transpose(v1))
# Repeat for v2, v3...
```

Saga 16 deliberately did NOT ship a `pca(X, k)`
builtin. The composition pattern is well-understood,
already demonstrated in a lesson, and any additional
wrapping would hide the mechanics that make PCA a
good thing to teach. If a concrete use case emerges
that demands a one-liner, the builtin becomes
trivial.

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
5. 3-D via column-selector matmul on the first three
   dims; render via `svg(..., "scatter3d")`.
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

## Why training-inside-a-chain is not shown

The natural demo shape for "learn embeddings, then
visualize" is:

```mlpl
model = chain(embed(V, d, 0), transformer_block, head)
train N { adam(cross_entropy(apply(model, X), Y), model, ...) }
table = apply(model.embed_layer, iota(V))  # <-- this
```

The last line is not possible today. `chain`'s child
evaluation goes through the generic `eval_expr` path,
which resolves bare identifiers only via `env.vars` and
not `env.models`. A pre-bound `emb_layer = embed(V, d,
0)` binding therefore can't be referenced inside a
`chain(...)` call. And there's no source-level way to
reach "the embed sublayer of this chain".

Workarounds:

- Use a standalone `embed` (no chain), train it
  directly. That is the demo's approach.
- Ship a language-level `embed_table(model) -> [V, d]`
  builtin that walks the ModelSpec and returns the
  table by internal name. Trivial to implement; not
  done in Saga 16 to keep scope tight.
- Extend `chain`'s child evaluation to fall through to
  `env.models` for bare idents. Also trivial; not done
  in Saga 16.

Either follow-up lands cleanly. The demo story works
as-is with the standalone variant.

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
- **`pca(X, k)` builtin.** PCA is composition-only
  today (power iteration + deflation, see Saga 8
  tutorial). Becomes trivial to add if the mechanics
  exposure ever stops being pedagogically useful.
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
- **`embed_table(model)` builtin.** Extract an embed
  layer's weights from a trained chain without using
  the `apply(standalone_embed, iota(V))` workaround.
  One-fn addition; not done in Saga 16.
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
- `contracts/viz-contract/scatter3d.md` -- 3-D
  scatter camera + rendering.
- `demos/embedding_viz.mlpl` -- CPU demo.
- `docs/demos-scripts.md` Demo 9 -- run guide.
- Saga 8 "Dimensionality Reduction: PCA" tutorial
  lesson -- PCA composition pattern.
- Sibling docs: `docs/using-perturbation.md`
  (Saga 20), `docs/using-lora.md` (Saga 15).
