# Experiment-Quality: design for evaluation rigor

Status: SHIPPED 2026-08-05 (saga experiment-quality). All three
builtins landed with tests and docs (`pareto_front`,
`param_count`, `experiment_metric`), plus `pareto_plot` -- the
staircase frontier renderer added on user request -- and the
three acceptance demos (Experiment Quality category). Review
resolutions: explicit direction vector for `pareto_front(P,
dirs)`; `experiment_metric` skips runs that lack the metric;
latency/RAM/energy remain externally measured (no in-language
stopwatch -- wasm has no clock and timing breaks demo replay).

## The thesis

A benchmark number without a robustness check and a cost axis is
an anecdote. Before the MTP / small-model program draws any
conclusion, the platform needs two facilities:

1. **Robustness suites** -- the same trained model evaluated
   across systematic perturbations of its task, so "it learned
   the task" and "it memorized the format" become distinguishable
   measurements.
2. **Pareto frontier analysis** -- experiment runs plotted on
   quality-vs-cost axes with the efficient frontier computed
   natively, so "is the bigger model worth it" is a picture, not
   a vibe.

Both build directly on shipped substrate: the `experiment` block
already records `*_metric` scalars and param shapes per run;
data-forge's `compress` / `grade_*` / `gather_rows` / `kg_*` are
the selection and task machinery; `scatter_labeled` and `svg`
are the renderers.

## Design constraints

1. **Array-first tasks.** Perturbations operate on `[n, L]` id /
   number arrays. Paraphrase-style text perturbations wait for
   the strings work; the design covers their array-shaped
   analogues (field reorder, identifier relabel, irrelevant
   context, hop extension, scaffold removal, input shift).
2. **Deterministic and replayable.** Every variant derives from
   the base set by a seeded or exact transform. No wall-clock
   dependence inside the language: quality, parameter count, and
   dataset size are the in-language Pareto axes. Latency, RAM,
   and energy are measured OUTSIDE the language today
   (docs/benchmarks.md methodology; serve-side telemetry); an
   in-language stopwatch is explicitly rejected for now because
   the wasm target has no clock and timing nondeterminism would
   break demo replay.
3. **Reuse before invent.** Perturbations are in-language idioms
   built from existing builtins wherever possible; new builtins
   only where composition genuinely cannot express the thing.

## Proposed new builtins (three, all small)

| Builtin | Semantics |
|---|---|
| `pareto_front(P, dirs)` | `[n, k]` metric matrix + `[k]` direction vector (`1` = maximize the column, `-1` = minimize) `->` `[n]` mask of non-dominated rows. Composes with the data-forge substrate: `compress(mask, P)` keeps the frontier, `scatter_labeled(P2, mask)` renders it highlighted. O(n^2) scan -- fine at experiment scale. |
| `param_count(m)` | Total trainable parameters of a model -- the x-axis of the quality-vs-size frontier. One number the user currently cannot get without hand-summing shapes. |
| `experiment_metric("name")` | `[runs]` vector of one recorded metric across the in-memory experiment log, in run order. The bridge from `experiment` blocks to arrays: column-concat several calls into the `[n, k]` matrix `pareto_front` eats. Missing metric in a run -> that run is skipped (the vector answers "every run that recorded this"). |

Placement: `pareto_front` in `mlpl-runtime-array` (a general
array algorithm); `param_count` and `experiment_metric` at the
eval layer (models and the experiment log live there).

## What stays in-language (deliberately)

The perturbation transforms are idioms, documented and used in
the demos -- each is one line of existing builtins:

- **Field reorder**: column permutation,
  `Xr = matmul(X, [[0, 1], [1, 0]])` (swap operands).
- **Identifier relabel**: send every id through a permutation
  table, `relabeled = reshape(gather_rows(reshape(perm, [k, 1]),
  flatten(T)), shape(T))` -- tests whether a model learned graph
  STRUCTURE or memorized specific ids. `grade_up(perm)` is the
  inverse permutation for free.
- **Irrelevant context**: concat a distractor column,
  `Xd = concat(X, reshape(rand_ints(n, 0, 10, s), [n, 1]) / 10, 1)`
  (model must be retrained with matching input width, or the
  distractor replaces a scaffold column -- see below).
- **Hop extension**: `kg_paths(edges, hops + 1, n, s)` -- the
  same generator, one notch harder.
- **Scaffold present/absent**: train with a hint column, then
  evaluate with the hint zeroed:
  `Xno = matmul(Xs, [[1,0,0],[0,1,0],[0,0,0]])`. The
  quality gap IS the scaffold-dependence measurement.
- **Input shift**: `X * 1.2`, `X + 0.3` -- classic covariate
  shift at whiteboard scale.
- **The suite runner**: evaluate one metric per variant, collect
  with `concat`, render with `svg(accs, "bar")`. A suite is a
  bar chart plus a convention, not a framework.

## Worked examples (the acceptance demos)

Robustness suite over the arithmetic classifier:

```text
m = chain(linear(2, 48, 0), relu_layer(), linear(48, 19, 1))
train 240 { adam(cross_entropy(apply(m, X), Y), ...); ... }
acc  = mean(eq(predict_batch(m, X), Y))            # baseline
accs = mean(eq(predict_batch(m, matmul(X, SWAP)), Y))  # operands swapped
accsh = mean(eq(predict_batch(m, X * 1.2), Y))     # input shift
svg(concat(...all variants...), "bar")             # the suite
```

Pareto frontier over widths:

```text
experiment "w4"  { ...train width-4...  loss_metric = ...; params_metric = param_count(m) }
experiment "w64" { ...train width-64... loss_metric = ...; params_metric = param_count(m) }
P = concat(reshape(experiment_metric("params_metric"), [n, 1]),
           reshape(experiment_metric("loss_metric"),   [n, 1]), 1)
front = pareto_front(P, [-1, -1])       # minimize both params and loss
scatter_labeled(P, front)               # frontier highlighted
```

Three demos, one new "Experiment Quality" category:

1. **Robustness Suite** -- train once, evaluate across five
   variants, bar-chart the accuracies; the takeaway names which
   drops are expected (swap hurts: the model never saw b + a)
   and which are benign.
2. **Scaffold Dependence** -- train WITH a hint column that
   leaks the answer's magnitude, evaluate with and without it;
   the visible cliff is why scaffold-absent evaluation must
   accompany every scaffolded training claim.
3. **Pareto Frontier** -- five widths trained in `experiment`
   blocks, quality-vs-parameters scatter with the frontier
   highlighted; diminishing returns become visible as the
   frontier flattens.

Every demo line gets the visual-verification treatment: rendered
natively, SVG inspected, and a HOW TO CHECK IT WORKED sentence in
the takeaway.

## Saga steps

1. **design** (this document; pause for review)
2. **pareto-core** -- `pareto_front` in mlpl-runtime-array, TDD
3. **experiment-bridge** -- `param_count` + `experiment_metric`
   at the eval layer, TDD
4. **demos** -- the Experiment Quality category (three demos),
   visual verification, pages deploy
5. **close** -- lang-reference + glossary + queue advance + wiki
   errata sync

## Open questions for review

1. `pareto_front(P, dirs)`: is the explicit direction vector
   right, or should the convention be "negate columns you want
   minimized" with no second argument? (Explicit dirs proposed:
   the call site documents itself.)
2. `experiment_metric` skips runs missing the metric. The
   alternative (error on a gap) is stricter but makes mixed logs
   unusable. Skip proposed.
3. Latency/RAM/energy axes: confirmed out of language scope for
   now (external measurement per benchmarks.md), revisited when
   serve-side telemetry can feed metrics back as first-class
   values?
