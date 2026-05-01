# First-class Distributions Milestone (Saga 24)

## Why this exists

Distributions are pedagogically central to ML and structurally
absent from MLPL today. A student learning variational methods,
mixture models, or RL policies has nothing to point at. The Saga
13 `sample(logits, t, seed)` builtin samples from an *implicit*
categorical distribution -- there is no value the student can
inspect, name, plot, or pass around.

Saga 24 introduces `Value::Distribution` as a first-class runtime
value with `Categorical`, `Gaussian`, and `Mixture` variants, plus
the operations that make distributions useful: `sample`,
`log_prob`, `entropy`, `kl_divergence`. This builds directly on
the Saga 23 `Probability` and `LogProbability` tags -- a
distribution's parameters are typed, and the result of `sample`
inherits the appropriate downstream tags.

Goal ranking applied:

- **Educational** is the leading goal. Distributions unlock VAE,
  mixture density network, policy-gradient, and Bayesian linear
  regression demos. Each becomes one or two web REPL lessons.
- **Correctness** follows: a distribution carries its parameters'
  shape and validity invariants (categorical sums to 1, Gaussian
  has positive variance), so misuse is caught at construction.
- **Extensibility**: the Distribution variant set is small and
  curated; user-defined distributions ship later (or never -- a
  composition of `Mixture` over `Gaussian` covers a lot).

## Non-goals

- Distributions over continuous domains beyond Gaussian. Beta /
  Dirichlet / Exponential are good follow-ups; ship them in a
  Saga 24.5 if the user demand emerges.
- Reparameterization gradients. The first version supports
  pathwise gradients through `sample` only for `Gaussian` (the
  classic VAE reparam trick); other variants raise a clean error
  inside `grad(...)`.
- Multivariate distributions with full covariance. `Gaussian`
  ships diagonal-covariance only.
- Conjugate posterior updates / inference. This is a primitives
  saga, not an inference saga.
- Distributions over distributions (hierarchical Bayes). Out of
  scope.
- New backend work. CPU only; MLX dispatch is a separate
  follow-up step under the existing Saga 14 / R1 frame.

## Quality requirements (every step)

Identical to Saga 23.

## What already exists

- `Value::Array` / `Value::Model` / `Value::Str` / (Saga 23)
  `ValueTag` machinery -- the precedent for adding a new
  `Value::Distribution` variant.
- `sample(logits, t, seed)` (Saga 13) -- the direct ancestor of
  `Categorical.sample()`. Saga 24 generalizes it.
- `randn(seed, shape)` (Saga 8) -- the Gaussian sampler primitive
  the new `Gaussian.sample()` reuses.
- `softmax` / `log_softmax` (Saga 6) -- the bridge between
  `Logit` and a `Categorical` constructor.
- Saga 23 tag machinery for typed parameters and typed outputs.

## Phases

### Phase 1: Distribution value variant

- New `Value::Distribution(DistributionSpec)` variant in
  `mlpl-eval`.
- `DistributionSpec` enum in a new
  `crates/mlpl-runtime/src/distributions.rs`:
  - `Categorical { probs: DenseArray }` -- rank-2 `[batch,
    classes]` of probabilities; row-sum invariant verified at
    construction.
  - `Gaussian { mean: DenseArray, std: DenseArray }` -- broadcast
    rules between mean and std identical to elementwise ops.
  - `Mixture { weights: DenseArray, components:
    Vec<DistributionSpec> }` -- weights sum to 1, all components
    same kind.
- `:describe d` for a `Distribution` prints the variant, parameter
  shapes, parameter previews, and the tag `Distribution`.
- Zero behavior change in any existing demo.

### Phase 2: Distribution constructors

Three constructor builtins:

- `categorical(probs)` -- accepts a `Probability`-tagged or
  untagged rank-2 array; verifies row-sum invariant.
- `categorical_from_logits(logits)` -- accepts a `Logit`-tagged
  array; internally applies softmax. The `from_logits` form is
  the recommended path because it is numerically stabler and the
  tag propagation is cleaner.
- `gaussian(mean, std)` -- elementwise broadcast between mean
  and std; rejects non-positive std.
- `mixture(weights, components)` -- accepts a vector of weights
  and a heterogeneous list of distributions of the same variant.

### Phase 3: sample / log_prob / entropy

Three per-distribution operations:

- `sample(d, seed)` -- the existing
  `sample(logits, t, seed)` is generalized; the old form keeps
  working as sugar for `sample(categorical_from_logits(logits)
  with temperature=t, seed)`. New form supports every variant.
  Output tag: integer index for `Categorical`, real-valued
  array for `Gaussian`, weighted draw for `Mixture`.
- `log_prob(d, x)` -- returns a `LogProbability`-tagged array
  whose shape matches `x`'s broadcast against `d`'s parameter
  shape.
- `entropy(d)` -- closed-form for `Categorical` and `Gaussian`,
  Monte-Carlo with explicit `n_samples` for `Mixture`.

### Phase 4: kl_divergence

- `kl_divergence(p, q)` over typed Distributions. Closed-form for:
  - `Categorical || Categorical`
  - `Gaussian || Gaussian` (diagonal)
- Mixed-variant KL raises a clean error with a hint pointing at
  Monte-Carlo as a future feature.
- Output tag: `Loss { kind: KLDivergence }` so the result threads
  into the optimizer pipeline as a regularizer.

### Phase 5: Reparameterization gradient for Gaussian

- `grad(loss, w)` where `loss` flows through
  `sample(gaussian(mean, std), seed)` rewrites the sample on the
  tape as `mean + std * eps` (eps = `randn(seed)` materialized
  once), enabling pathwise gradients.
- `Categorical.sample` inside `grad(...)` raises a clean error
  with a hint suggesting the Gumbel-softmax follow-up (deferred).

### Phase 6: Visualization

- `svg(d, "distribution")` -- per-variant rendering:
  - `Categorical` -> bar chart of class probabilities.
  - `Gaussian` -> 1-D bell curve (or grid of bell curves for
    rank-2 mean/std).
  - `Mixture` -> overlaid components with their weights.
- `svg(d, "samples", n)` -- draw `n` samples and plot them.
  Histograms for `Gaussian`, stacked bars for `Categorical`.

### Phase 7: Demos

- `demos/vae_mnist.mlpl` -- VAE on a tiny MNIST-shaped synthetic
  dataset (4x4 images, 100 samples). Encoder produces a Gaussian
  posterior, decoder produces a Categorical reconstruction,
  KL-divergence to a Gaussian prior is the regularizer. End-to-end
  trains in <30 seconds on CPU.
- `demos/policy_gradient.mlpl` -- a tiny REINFORCE policy on a
  bandit problem. Categorical policy, log_prob in the loss,
  sample in the action.
- `demos/mixture_density.mlpl` -- a 1-D mixture density network
  predicts a multimodal target with a Mixture(Gaussian).

### Phase 8: Tutorial lessons + retrospective + release

- Three new web REPL lessons:
  - "Distributions Basics" -- construct, sample, plot.
  - "Variational Autoencoders" -- VAE walkthrough.
  - "Policy Gradient" -- REINFORCE walkthrough.
- `docs/using-distributions.md` retrospective + user guide.
- Update `docs/saga.md`, `docs/status.md`,
  `docs/are-we-driven-yet.md`.
- Bump REPL banners; rebuild `pages/`; tag the release.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | distribution-value-variant   | 1 | `Value::Distribution` + `DistributionSpec` + `:describe` |
| 002 | distribution-constructors    | 2 | `categorical`/`categorical_from_logits`/`gaussian`/`mixture` |
| 003 | sample-logprob-entropy       | 3 | three per-distribution ops; old `sample` form preserved |
| 004 | kl-divergence                | 4 | closed-form KL for Cat \|\| Cat and Gauss \|\| Gauss |
| 005 | reparam-gradient-gaussian    | 5 | pathwise gradients through Gaussian.sample |
| 006 | distribution-visualization   | 6 | `svg(d, "distribution")` and `svg(d, "samples", n)` |
| 007 | vae-policy-mixture-demos     | 7 | three new headline demos |
| 008 | distributions-tutorials      | 8 | three new web REPL lessons |
| 009 | distributions-release        | 8 | docs, banners, pages rebuild, release tag |

Nine steps. The reparam-gradient phase (005) may slip to a
follow-up if the autograd tape integration grows beyond a single
step's budget.

## Success criteria

- `gaussian(0.0, 1.0)` constructs, `:describe` shows
  `Gaussian(mean=0, std=1)` plus the `Distribution` tag.
- `sample(gaussian(0, 1), seed)` returns a scalar that
  `:describe` reports as derived from the Gaussian.
- `log_prob(gaussian(0, 1), 0.0)` equals the textbook
  -0.5 * log(2 * pi) within fp64 tolerance.
- `kl_divergence(categorical(p), categorical(q))` for `p == q`
  is zero; for `p != q` matches the textbook formula.
- `demos/vae_mnist.mlpl` trains end-to-end and the reconstruction
  KL term decreases monotonically over 100 Adam steps.
- The "Distributions Basics" lesson runs in the browser without
  external network access.
- All existing demos still pass.
- Quality gates green; pages deployed; release tagged.

## Risks and open questions

- **Categorical row-sum invariant verification.** Constructed
  from `categorical_from_logits` it is exact (modulo fp); from
  raw probs the user might pass an array that *almost* sums to 1.
  Policy: tolerance of 1e-5 per row, with a clean error pointing
  at `categorical_from_logits` as the recommended path.
- **Mixture component homogeneity.** First version restricts
  components to be the same variant. Heterogeneous mixtures (a
  Gaussian + a Categorical?) are nonsense semantically; reject
  cleanly at construction.
- **Reparam gradient bookkeeping.** The Gaussian sample sits on
  the tape; the tape's `eps` slot must be reused on every
  backward pass (re-sampling would change the loss surface).
  Document the policy in the autograd contract.
- **Mixture entropy via Monte Carlo.** Closed-form entropy doesn't
  exist for mixtures of Gaussians. Default to MC with `n_samples
  = 256` and a `with samples=N` syntactic sugar; revisit once the
  user has tried it.
- **MLX dispatch.** Categorical sampling on MLX needs the
  cumulative-sum + uniform-search pattern; Gaussian sampling
  reuses `randn`. Initial saga ships CPU-only; an MLX phase 9
  lands later if the demos warrant it.
