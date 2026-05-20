# Dimensionality Reduction Milestone

Proposed (saga number TBD; deliberately not numbered until the user
approves the plan).

## Why this exists

MLPL ships `pca(X, k)` and `tsne(X, perplexity, iters, seed)` as
builtins plus a single PCA demo. That is a tiny slice of the field,
and there is no tutorial coverage and no learning path. Users
arriving with high-dimensional data (an `[N, 768]` ViT patch
embedding, an `[N, D]` document-embedding matrix, an
`[N, hidden]` LLM activation dump) have no guided route from "this
matrix is too wide to look at" to "here is a chart that shows what
matters."

This milestone closes that gap. Goals, in order:

- **Educational.** The ladder runs from "why reduce dimensions at
  all" through PCA -> t-SNE -> UMAP -> a brief tour of other
  methods, each lesson introducing one new idea.
- **Visual.** Every method ships with at least one demo. The
  headline visual is a *critical-dimensions heatmap*: a per-feature
  importance map that tells the user *which input dimensions
  matter*, not just where the points end up after reduction.
- **Correctness.** PCA's existing power-iteration deflation passes
  the parity test today; t-SNE has been gradchecked. New methods
  must hold the same standard.
- **Utility.** All methods runnable from REPL + web playground + on
  the MLX peer where applicable.
- **Performance.** Not a goal at this layer. Big-data variants
  (Faiss-style approximate nearest neighbors, GPU UMAP) are
  deferred.

## Non-goals

- **Large-scale data.** Target sizes are `N <= 5000`, `D <= 1024`.
  At those sizes a tree-walked Rust implementation is fast enough;
  approximate nearest-neighbor structures (HNSW, IVF, etc.) are
  out.
- **GPU acceleration of any new method.** PCA and t-SNE stay CPU
  today; UMAP and MDS will too. The MLX peer track picks this up
  if a demo demands it.
- **All known dim-reduction methods.** Isomap, Laplacian
  Eigenmaps, Diffusion Maps, Locally Linear Embedding, Sparse PCA,
  Kernel PCA, ICA, NMF, autoencoder-based reduction -- all
  deferred. The included set covers ~90% of what a learner needs.
- **Cluster auto-detection.** HDBSCAN, k-means-on-the-embedding,
  silhouette scoring, etc., are visualization companions, not
  dim-reduction primitives. They can come in a sibling milestone.

## Dependencies

- **Value::Result** (Saga 29 step 012): UMAP's neighbor-graph
  construction can fail on degenerate input; surface that via
  Result.
- **String list** (Saga 29 step 002): pass feature names through
  to the critical-dimensions heatmap so axis labels are
  meaningful.
- **`gather` / slice ranges** (audit #12): UMAP and the
  critical-dimensions viz both want non-contiguous row selection;
  if this lands first the implementations are cleaner. Not a
  hard block -- a `take` loop works, just verbosely.

## What already exists

- `pca(X, k)`: top-k PCA via power iteration + deflation
  (`crates/mlpl-runtime/src/pca_builtin.rs`). Returns centered,
  projected `[N, k]`.
- `tsne(X, perplexity, iters, seed)`: t-SNE 2D embedding
  (`crates/mlpl-runtime/src/tsne_*.rs`). Returns `[N, 2]`.
- `demos/pca.mlpl`: one demo, builds a correlated 2D dataset and
  recovers the principal axis by hand.
- Glossary entries: **PCA (Principal Component Analysis)**,
  **t-SNE**, **Manifold Hypothesis**, **Embedding**.
- Existing viz: `scatter_labeled`, `scatter3d`, `heatmap`,
  `heatmap_grid`. All composable.

## Quality requirements (every step)

Identical to Saga 29. TDD; four `cargo` gates +
`markdown-checker` + `sw-checklist` green; `/mw-cp` checkpoint;
push after every commit; web changes rebuild `pages/`; `.agentrail/`
committed.

New builtins ship with at minimum a unit test on a small fixture
and an integration test against a known-correct reference (sklearn
output for PCA / UMAP / MDS, captured as a static fixture --
*not* a sklearn dependency at build time).

## Phases

### Phase 1: critical-dimensions heatmap (visual primitive)

This is the headline visual and the highest-leverage step.
PCA today returns the *projected* data but throws away the
*components* (the loadings: which input dimensions contributed to
each output dimension). UMAP and MDS do not have direct
"loadings" but DO have feature-importance via permutation
sensitivity.

Two phases:

1. **`pca_components(X, k)`** -- return `[k, D]` matrix where row
   `i` is the i-th principal component (the direction in original
   feature space). Today the components are computed inside
   `pca` but discarded. Surface them.

2. **`svg(loadings, "critical_dimensions", feature_names)`** --
   new viz type. Takes a `[k, D]` loadings matrix and an optional
   `[D]` string list of feature names. Renders a horizontal
   heatmap with `D` columns and `k` rows; cell `(i, j)` colored by
   the magnitude of feature `j`'s contribution to component `i`.
   The top-N most "critical" features per component are
   highlighted (e.g. bordered or marked). Right-side legend has
   per-component variance-explained percentages.

The output reads as: "Component 1 is mostly pixel-row-32 and
pixel-row-33. Component 2 is mostly pixel-row-0 and pixel-row-63.
The first two components together explain 84% of the variance."

For t-SNE and UMAP the same viz is run on a *permutation
sensitivity* matrix: for each input dimension, shuffle it across
samples and measure the increase in the embedding's stress /
reconstruction error. The dimensions whose shuffling most
disturbs the embedding are the "critical" ones.

### Phase 2: UMAP (the headline implementation)

UMAP (Uniform Manifold Approximation and Projection, McInnes and
Healy 2018) is the practical winner the demos point at. The
implementation is non-trivial -- it builds a k-NN graph, computes
fuzzy-set membership, and optimizes a low-D layout via stochastic
gradient descent on a cross-entropy objective. The motivation
section of every UMAP lesson and demo positions it against PCA
(linear, fast, throws away non-linear structure) and against
*SNE methods (described in phase 4, runnable comparison only via
the existing `tsne` builtin).

Steps (each is its own agentrail step):

1. **k-NN graph builder** (`knn_graph(X, k)` returning a sparse
   edge list). Simple brute-force for now -- the size budget is
   small.
2. **Fuzzy simplicial set construction** -- the math is local
   sigma estimation + symmetrization.
3. **Layout optimization** -- 2D coordinate SGD on the
   cross-entropy + repulsion objective.
4. **`umap(X, n_neighbors, min_dist, iters, seed)` builtin** that
   wraps the above into one call. Returns `[N, 2]`.

### Phase 3: comparison demos (UMAP as the winner)

These are the demos the milestone is built around. Each is a
side-by-side that ends with "UMAP got the structure the other
method missed."

1. **`demos/umap_vs_pca.mlpl`** -- a curved manifold dataset
   (Swiss roll or two-moons embedded in higher D). PCA's linear
   projection unrolls into a smear because the principal axis
   crosses the manifold; UMAP recovers the original geometry.
   Renders both 2D embeddings side by side.

2. **`demos/umap_vs_tsne.mlpl`** -- a multi-cluster dataset where
   inter-cluster distance is meaningful (e.g. three clusters at
   distinct radii from origin). t-SNE preserves local structure
   but distorts global distances; UMAP preserves both. Each
   embedding gets a colored scatter; the legend calls out which
   global feature each method preserved.

3. **`demos/dim_reduction_zoo.mlpl`** -- the full lineup: PCA,
   t-SNE, UMAP, MDS, random projection on one dataset, rendered
   as a row of five thumbnails with a caption per method. Pure
   browse-and-compare reference. (Builds on the additional
   methods in Phase 5.)

The demos prefer UMAP for the obvious-winner cases; the lessons
in Phase 4 cover the cases where PCA or t-SNE is the right
choice.

### Phase 4: tutorial lessons (SNE family described, UMAP demoed)

Lessons added to the tutorial track in dependency order:

1. **"Why reduce dimensions?"** -- motivation lesson. The
   manifold hypothesis; the curse of dimensionality; visualizing
   high-D data. No code; concept-first.

2. **"PCA: the linear baseline"** -- builds on `demos/pca.mlpl`,
   adds the loadings view via `pca_components` + `svg(_,
   "critical_dimensions")`. Positioned as "the fastest answer,
   the one to try first."

3. **"SNE: the very-slow ancestor"** -- *description-only,
   no MLPL primitive*. Walks through Hinton and Roweis 2002:
   gaussian similarity in both high-D and low-D, asymmetric KL
   divergence, the symmetrization-and-crowding problem that made
   it slow to converge and prone to collapsing clusters. The
   lesson includes the equations + a "why it was abandoned"
   framing; it does NOT propose adding an `sne` builtin since
   t-SNE strictly dominates.

4. **"t-SNE: a peek at nonlinear methods"** -- the existing
   `tsne` builtin. Lesson explains van der Maaten and Hinton
   2008's two fixes (symmetric KL + Student-t in low-D) and
   notes the Barnes-Hut 2014 follow-up that brought it to
   O(N log N). The standard "cluster shape is meaningful,
   distances between clusters are not" caveat. Used in
   comparison demos but never as the recommended default.

5. **"UMAP: the modern default"** -- the headline lesson.
   Introduces UMAP from first principles (Riemannian geometry
   + fuzzy simplicial sets, framed informally), runs the
   `demos/umap_vs_pca.mlpl` and `demos/umap_vs_tsne.mlpl`
   demos, and explains *why* UMAP preserves more global
   structure than t-SNE: the cross-entropy objective penalizes
   both attractive and repulsive forces evenly, while t-SNE's
   KL is mostly attractive.

6. **"Reading a critical-dimensions heatmap"** -- a lesson
   centered on the viz itself. Shows the same data viewed
   through PCA loadings, permutation sensitivity for t-SNE,
   and permutation sensitivity for UMAP; teaches the user to
   read each.

### Phase 5: additional methods (one builtin per step)

These are the "complete the survey" methods. None of them get a
demo of their own -- they show up in `demos/dim_reduction_zoo.mlpl`
as side-by-side comparisons against UMAP.

1. **MDS (Multidimensional Scaling)** -- `mds(X, k, iters)`. Pre-
   computes the pairwise distance matrix and finds low-D
   coordinates that preserve those distances. Classical MDS is
   eigendecomposition; metric MDS is SGD. Ship the SGD variant
   (simpler, reuses t-SNE's optimization loop). Briefly
   mentioned in the UMAP lesson as "the geometric ancestor."

2. **Random Projection** -- `random_projection(X, k, seed)`. The
   Johnson-Lindenstrauss surprising result: a random `[D, k]`
   projection approximately preserves pairwise distances for
   modest `k`. One-liner implementation; the payoff is the
   "you do not always need principled methods" lesson, which
   gets a short section inside the "Why reduce dimensions?"
   lesson rather than its own.

Optional steps (added only if a user demand surfaces):

- **Kernel PCA** -- needs a kernel matrix; the natural way is to
  expose `kernel_matrix(X, "rbf", gamma)` first.
- **ICA** -- needs whitening + a contrast function. Useful for
  signal separation but a niche introduction.
- **Barnes-Hut t-SNE** -- O(N log N) approximation of t-SNE via
  spatial trees. Useful if MLPL ever wants to demo t-SNE on
  larger N; deferred until a real demand surfaces.

### Phase 6: learning path

New path **"High-dimensional data, one chart at a time"** in
`apps/mlpl-web/src/paths.rs`. Walks PCA -> SNE family ->
UMAP, then loops back through the critical-dimensions viz:

- Note: "Why this path exists"
- Glossary: **Manifold Hypothesis**
- Glossary: **Embedding**
- Lesson: "Why reduce dimensions?"
- Diagram: a new SVG showing the conceptual workflow:
  raw `[N, D]` -> dim reduction -> `[N, 2]` for visualization +
  `[k, D]` critical-dimensions view -> insight.
- Lesson: "PCA: the linear baseline"
- Demo: existing `demos/pca.mlpl`
- Glossary: **PCA**
- Lesson: "SNE: the very-slow ancestor" (description-only)
- Glossary: **SNE** (new entry)
- Lesson: "t-SNE: a peek at nonlinear methods"
- Glossary: **t-SNE**
- Lesson: "UMAP: the modern default"
- Glossary: **UMAP** (new entry)
- Demo: `demos/umap_vs_pca.mlpl`
- Demo: `demos/umap_vs_tsne.mlpl`
- Lesson: "Reading a critical-dimensions heatmap"
- Demo: `demos/dim_reduction_zoo.mlpl` (the full lineup)
- Note: "Beyond this path" -- pointer to deferred methods
  (Isomap, LLE, Kernel PCA, autoencoder embeddings,
  Barnes-Hut t-SNE).

## Editorial stance (confirmed 2026-05-20)

User direction: focus demos on UMAP being "better" than PCA and
the *SNE family; SNE itself is description-only (no builtin --
it is too slow to be useful and t-SNE dominates anyway), and
t-SNE is described in detail but only appears in
demos as the loser in head-to-head comparisons against UMAP.
That framing is baked into Phase 3 (UMAP-vs-X demos), Phase 4
(SNE described-only, t-SNE described + comparison-only), and
Phase 6 (path ordering puts PCA -> SNE -> t-SNE -> UMAP as the
historical-progression spine).

## What I want to confirm before starting

- Whether **Phase 1 (critical-dimensions heatmap)** should ship
  ahead of UMAP. It can land on top of the existing `pca`
  builtin alone; the new viz is the highest-leverage piece and
  gives the dim-reduction track a visible win before the UMAP
  implementation lands. Default plan is yes -- phase 1 first.
- Whether the `demos/umap_vs_pca.mlpl` and
  `demos/umap_vs_tsne.mlpl` demos should each include a
  critical-dimensions heatmap of their input data (so the
  reader sees what UMAP found AND which input dimensions
  carried the signal), or whether that should stay in the
  zoo demo only. Default: include in both -- the viz is the
  whole point.
- Whether the **chronological-history milestone** (the
  sibling proposal) should pull these history-of-DR lessons
  in by reference (one entry "SNE -> t-SNE -> UMAP" in its
  spine) rather than duplicating content.
- The order of phase 4 (lessons) and phase 5 (path) -- I have
  them sequential here, but they can interleave: write the
  path's skeleton early and fill in lessons as each phase
  lands.
