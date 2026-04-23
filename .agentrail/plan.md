# Embedding Visualization Milestone (Saga 16, v0.14.0)

## Why this exists

After Saga 15, MLPL can train a base, LoRA-fine-tune it on
a custom dataset, and watch the loss curve. The next
natural move is "what did the model actually learn?" -- and
for any model with an `embed` layer or an intermediate
representation, that question is answered with
visualization. Plot the embedding table after training;
color by cluster; pick the k nearest neighbors of a query
token and see whether they are semantically related.

Saga 16 adds the primitives and viz types for that
workflow. Three new builtins (`pairwise_sqdist`, `knn`,
`tsne`) plus one new viz type (`svg(..., "scatter3d")`)
compose with the existing `embed` / `apply` / `svg`
surface to give "train -> inspect embeddings -> render"
in a handful of MLPL lines.

## Non-goals (deferred)

- **RAG pipeline demo.** Real RAG needs a retrieval -> LLM
  prompt augmentation flow; without a local inference path
  (Saga 19's LLM-as-tool sidecar or a checkpoint format)
  the demo is performative. Sibling saga once Saga 19
  ships.
- **UMAP.** UMAP is a second nonlinear reducer that
  overlaps heavily with t-SNE in user experience;
  shipping both in one saga doubles the testing surface
  with marginal user value. Defer until there is concrete
  demand.
- **PCA as a new builtin.** PCA is already expressible in
  MLPL today via power iteration on a covariance matrix
  (see the "Dimensionality Reduction: PCA" tutorial
  lesson, Saga 8). We will document PCA as a composition
  pattern rather than a builtin.
- **MLX dispatch for `tsne`.** t-SNE's inner loop
  (per-point perplexity binary search, gradient descent
  over pairwise affinities) is irregular and would need
  careful MLX-specific lowering to be faster than CPU at
  embedding-dataset scale. Defer; `pairwise_sqdist` and
  `knn` themselves route through existing matmul /
  reduction dispatch so they inherit MLX for free.
- **Live interactive 3D.** WebGL / rotation / zoom. The
  SVG-projected 3D scatter is a static snapshot good for
  docs and for "is my cluster structure real?" checks.

## Quality requirements (every step)

Identical to Saga 15:

1. TDD: failing test first, then implementation, then
   refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.
6. `.agentrail/` changes are committed whenever they
   change.
7. New FAILs in `sw-checklist` must be resolved in the
   same step (extract modules / use structs for many
   args).

## What already exists

- Saga 7's viz stack: `svg(data, type)` with `scatter`,
  `line`, `bar`, `heatmap`, and analysis helpers
  (`scatter_labeled`, `hist`, `loss_curve`,
  `confusion_matrix`, `boundary_2d`). Lives in
  `crates/mlpl-viz`.
- Saga 8's PCA via power iteration (demos and tutorial
  lesson). Not shipped as a builtin but expressible
  today.
- Saga 13's `embed(vocab, d_model, seed)` + Saga 15's
  LoRA on embedding layers.
- K-Means demo pattern for pairwise-squared-distance via
  `matmul` + `reduce_add` + broadcasting (reused in
  step 001's TDD).

## Phase 1 -- distance + nearest-neighbor primitives (1 step)

### Step 001 -- `pairwise_sqdist(X)` + `knn(X, k)`
Two sibling builtins.

`pairwise_sqdist(X) -> D` takes rank-2 `X [N, D]` and
returns `[N, N]` where `D[i, j] = sum_k (X[i, k] -
X[j, k])^2`. Implementation composes the familiar
`matmul + reduce_add + broadcasting` identity. Shipping as
a builtin (not a demo pattern) gives us a single place for
the shape check and avoids the 3-line boilerplate every
demo currently re-types.

`knn(X, k) -> idx [N, k]` returns, for each row of `X`,
the indices of the `k` nearest rows by squared Euclidean
distance. Ties break by lower index (stable sort per
row). Row `i` itself is excluded from its own neighbor
list.

Contracts:
- `contracts/eval-contract/pairwise-sqdist.md`
- `contracts/eval-contract/knn.md`

TDD in `crates/mlpl-eval/tests/pairwise_sqdist_knn_tests.rs`:
- Hand-constructed 3-point fixture: shape checks, symmetry
  of D, zero diagonal.
- `knn(X, 1)` picks the nearest non-self neighbor for each
  of 4 points on a 1-D line.
- `knn(X, 2)` returns them sorted by distance (ascending).
- Self-exclusion: `i` never appears in `knn(X, k)[i]`.
- Error cases: rank-!=2 X, k = 0, k >= N, non-integer k.

Module placement: one new
`crates/mlpl-runtime/src/embedding_builtins.rs` (small,
under the 7-fn budget). Dispatched from
`crates/mlpl-runtime/src/builtins.rs` in the same style as
`ensemble_builtins` from Saga 20.

## Phase 2 -- t-SNE + 3D scatter (2 steps)

### Step 002 -- `tsne(X, perplexity, iters, seed)` builtin
Classic Laurens van der Maaten t-SNE: perplexity-
calibrated conditional probabilities in the high-dim
space, Student's-t affinities in the low-dim space, KL
divergence loss, gradient descent. Output is 2-D
(shape `[N, 2]`). Fixed hyperparameters beyond
user-controlled ones: learning rate 200 (van der Maaten
default), early exaggeration factor 4 for the first 100
iterations, momentum 0.5 ramping to 0.8. Document every
hyperparameter choice in the contract.

Implementation detail: ship as a pure-Rust builtin in
`crates/mlpl-runtime/src/embedding_builtins.rs` rather
than trying to express the inner loop in MLPL source. The
perplexity calibration is a per-point binary search that
does not vectorise cleanly through the autograd tape.

TDD in `crates/mlpl-eval/tests/tsne_tests.rs`:
- Fixture: three well-separated Gaussian blobs in
  `[60, 4]`. Run `tsne(X, 30.0, 200, 42)`; verify each
  cluster's 2-D centroid is distinct (pairwise distance
  between centroids > threshold). This pins the
  "structure is preserved" claim without asserting exact
  coordinates (t-SNE has rotational and reflection
  symmetry and is seed-sensitive).
- Output shape `[N, 2]` for input `[N, D]` (any D).
- Determinism: two calls with the same `(X, perplexity,
  iters, seed)` produce bit-identical output.
- Error cases: rank-!=2 X, perplexity >= N, iters < 1,
  non-finite X.

Contract: `contracts/eval-contract/tsne.md`.

### Step 003 -- 3D scatter viz (`svg(pts, "scatter3d")`)
New viz type in `crates/mlpl-viz`. Accepts rank-2 input
`[N, 3]` (points) or `[N, 4]` (points + cluster id in the
last column, mirroring `scatter_labeled`). Projects via a
fixed orthographic camera at a documented azimuth /
elevation (30 / 20 degrees is a reasonable default).
Renders axis gizmos plus the projected dots with the
standard color palette used by `scatter_labeled`.

Goals:
- Pretty enough for docs and "is my cluster structure
  real?" at-a-glance checks.
- Deterministic SVG output for snapshot testing.
- No interactivity; no rotation. That is a follow-up
  saga if anyone ever asks for it.

TDD in `crates/mlpl-viz/tests/scatter3d_tests.rs`:
- Shape validation.
- SVG string contains the expected number of `<circle>`
  elements (= N dots).
- Legend appears when a 4th column is present.
- Snapshot test against a small fixed fixture so
  regressions in the projection math are caught.

Contract: `contracts/viz-contract/scatter3d.md`.

## Phase 3 -- demo + docs + release (3 steps)

### Step 004 -- `demos/embedding_viz.mlpl`
End-to-end CPU demo:
1. Load the Shakespeare preloaded corpus + train a
   byte-level BPE (same setup as every Saga 13+ demo).
2. Build a small model with an `embed` layer; train
   briefly so the embedding actually learns structure.
3. Extract the `[vocab, d]` embedding table; apply `tsne`
   to reduce to 2-D.
4. Render `svg(emb_2d, "scatter")`.
5. Same extraction reduced to 3-D via an extra PCA-to-3
   projection of the embedding table; render
   `svg(emb_3d, "scatter3d")`.
6. For a sample query token, report `knn(emb, 5)` (the
   5 nearest tokens by embedding distance) with
   `apply_tokenizer` reverse lookup so the output is
   human-readable.

Integration test
`crates/mlpl-eval/tests/embedding_viz_tests.rs`:
- Cut-down fixture (small corpus, V=32, d=8, 3 train
  steps). Assert output shapes + finite values; assert
  `knn` indices are all in `[0, V)` and exclude the
  query itself.
- Renders do not panic (fixtures are tiny so the SVG is
  short).

Add a one-line Demo 9 entry to `docs/demos-scripts.md`.

### Step 005 -- tutorial lesson + `docs/using-embeddings.md`
Add an "Embedding exploration" lesson to
`apps/mlpl-web/src/lessons.rs` following the Saga 15 /
Saga 20 pattern: tiny interactive variant that fits the
500-LOC `lessons.rs` budget. Walk the user through
`pairwise_sqdist` on a hand-picked 4-point vector, `knn`
on the same, a small t-SNE reduction, and a 3-D scatter
render. `try_it` suggests changing the perplexity and
watching the layout.

Rebuild `pages/` via `./scripts/build-pages.sh` and
commit both source and the WASM bundle together.

`docs/using-embeddings.md` covers:
- What embeddings are and why we visualize them.
- The three new builtins (`pairwise_sqdist`, `knn`,
  `tsne`) with one-paragraph rationale each and a pointer
  to the contract.
- PCA as a composition pattern (link to Saga 8's tutorial
  lesson; note we did not ship a `pca` builtin and why).
- 3-D scatter surface.
- The shipped demo walkthrough.
- Deferred follow-ups: UMAP, `pca` builtin, RAG pipeline
  (pending Saga 19), interactive 3-D, MLX for t-SNE.

### Step 006 -- release v0.14.0
Bump workspace version 0.13.0 -> 0.14.0. Update
`CHANGELOG.md` with the v0.14.0 section. Insert a Saga 16
retrospective in `docs/saga.md` above Saga 15
(newest-first). Move Saga 16 row in `docs/status.md` from
Planned to Completed; roll remaining Planned target
versions forward. Tag v0.14.0 locally; confirm with user
before pushing the tag. `agentrail complete --done`
closes the saga.

## Dependency graph

```
001 pairwise_sqdist + knn
  \-- 002 tsne
        \-- 003 3D scatter viz
              \-- 004 embedding_viz demo
                    \-- 005 tutorial + using-embeddings.md
                          \-- 006 release v0.14.0
```

Steps are strictly sequential; each uses the prior steps'
surface as a fixture.
