# `tsne` Contract (Saga 16 step 002)

## Purpose

`tsne(X, perplexity, iters, seed) -> Y` is classic van der
Maaten t-SNE for dimensionality reduction: given a rank-2
`[N, D]` input, produce a rank-2 `[N, 2]` low-dimensional
embedding that preserves local neighborhood structure.
Used in the Saga 16 embedding-visualization demos to
render learned embedding tables as 2-D scatter plots.

## Signature

```
tsne(X, perplexity, iters, seed) -> Y
```

- `X` -- rank-2 float array `[N, D]`. All entries must be
  finite.
- `perplexity` -- positive f64. Must satisfy
  `0 < perplexity < N`. Typical range is 5-50; larger
  perplexity emphasizes global structure, smaller
  emphasizes local clusters. `N=60, perplexity=15` is a
  reasonable default for the demo-scale fixtures used
  in this saga.
- `iters` -- positive integer number of gradient-descent
  steps. 250-1000 is typical.
- `seed` -- f64 used to initialize `Y` via
  `randn(seed, [N, 2]) * 1e-4`.
- `Y` -- rank-2 float array `[N, 2]`.

## Algorithm

### High-dim conditional probabilities `P`

For each row `i`, binary-search beta_i such that the
Shannon entropy of
`P_{j|i} = exp(-beta_i * D2[i, j]) / Z_i` (self excluded)
equals `log(perplexity)`. Bisection uses up to 50
iterations or until `|H - log(perplexity)| < 1e-5`.

Symmetrize: `P_{ij} = (P_{j|i} + P_{i|j}) / (2 * N)`.
Clamp every entry to at least `1e-12` to avoid `log(0)`
in gradient computations.

### Low-dim affinities `Q`

`Q_{ij} = Q_{ij}^unnorm / Z` where
`Q_{ij}^unnorm = 1 / (1 + ||Y_i - Y_j||^2)` (Student-t
with one degree of freedom; self excluded) and `Z = sum_{i
!= j} Q_{ij}^unnorm`. Clamp each `Q_{ij}` to at least
`1e-12`.

### Gradient

`dY_i = 4 * sum_j (exag * P_{ij} - Q_{ij}) *
Q_{ij}^unnorm * (Y_i - Y_j)` where `exag` is the
early-exaggeration factor (see hyperparameters below).

### Update

- Learning rate 200.
- Momentum 0.5 for `iter < 250`, 0.8 afterwards.
- `update_t = momentum * update_{t-1} - lr * dY`; `Y +=
  update_t`.
- Center `Y` at the origin every step so the solution
  does not drift.

## Hyperparameters (van der Maaten defaults)

- `LEARNING_RATE = 200`
- `EARLY_EXAG = 4` for the first 100 iterations, 1
  afterwards
- `MOMENTUM` = 0.5 for the first 250 iterations, 0.8
  afterwards
- Initial `Y` scale: `randn * 1e-4`
- `BISECTION_ITERS = 50`; `BISECTION_TOL = 1e-5`
- `MIN_PROB = 1e-12` (clamp on both `P` and `Q`)

None of these are user-tunable in Saga 16; they are
baked into the builtin. A follow-up saga may expose them
via a config struct if there is concrete demand.

## Determinism

Two calls with identical `(X, perplexity, iters, seed)`
produce bit-identical `Y`. The only sources of
randomness are the Xorshift64 PRNG used for the initial
`Y` (seeded by the `seed` argument) and the sort order
of stable `f64` comparisons -- no NaN handling, no
thread-level non-determinism.

## Rotational + reflection ambiguity

The KL-divergence loss is invariant under arbitrary 2-D
rotations and reflections of `Y`, so the absolute
coordinates returned by `tsne` are seed-sensitive and
NOT reproducible across seeds. Tests pin
**structure-preservation** claims (cluster centroids
pairwise-distinct) rather than exact coordinates.

## Error cases

All errors surface as
`RuntimeError::InvalidArgument { func: "tsne", reason }`
(or `ArityMismatch` for wrong arity), propagated as
`EvalError::RuntimeError(...)`.

- **Wrong arity.** Anything other than 4 arguments.
- **Non-rank-2 `X`.** "X must be rank-2 [N, D], got rank
  N".
- **Non-finite `X` entries.** "X must be finite; element
  i is <value>". Caller must filter NaN / infinity
  before calling.
- **Non-scalar `perplexity` / `iters` / `seed`.** "<name>
  must be a scalar".
- **Non-positive `perplexity`.** "perplexity must be
  positive, got <value>".
- **`perplexity >= N`.** "perplexity <value> must be <
  N = N; the binary search cannot find a beta that
  matches perplexity >= N".
- **Non-positive or non-integer `iters`.** "iters must
  be a positive integer, got <value>".

## What this contract does NOT cover

- **UMAP.** Separate sibling algorithm; not shipped in
  Saga 16 (overlaps with t-SNE in user experience for
  marginal value; doubling the testing surface was out
  of scope).
- **3-D output (`t-SNE -> [N, 3]`).** The builtin only
  produces 2-D; for 3-D embedding visualization, use
  PCA via power iteration on the embedding covariance
  matrix (see Saga 8 tutorial lesson). 3-D t-SNE is a
  trivial follow-up once a use case surfaces.
- **MLX dispatch.** t-SNE's inner loop (per-point
  perplexity binary search, `P`/`Q` computations) does
  not vectorize cleanly through MLX's kernel model at
  embedding-table scale. CPU-only today. Deferred.
- **Exposed hyperparameters.** `LEARNING_RATE`,
  `MOMENTUM_SWITCH`, `EARLY_EXAG` are baked. A config-
  struct API is a follow-up.
- **Approximate / Barnes-Hut acceleration.** The exact
  `O(N^2)` algorithm is fine at embedding-table scale;
  approximate methods land once anyone runs t-SNE on
  `N >= 10000`.
