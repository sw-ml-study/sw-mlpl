# Optional Typing for MLPL (design)

**Status:** design proposal, 2026-05-01. Seeds Saga 23 (typed
values + typed traces) through Saga 28 (user-defined tags). Builds
directly on `docs/typing.md` (today's untyped surface),
`docs/milestone-named-axes.md` (Saga 11.5 labeled axes), and
`docs/typed-ml-concepts.md` (Tier A/B/C ML-concept proposal). This
document supersedes `docs/typed-ml-concepts.md` as the umbrella
plan; the Tier A/B vocabulary from that doc is rolled into the
sagas below.

For project-level goal ranking, this doc treats:

> educational > correctness > utility > practicality
> > maintainability > extensibility > ... > performance

as a load-bearing constraint. Every design choice below should be
re-derivable from that ranking.

## Why now

MLPL today is dynamically and weakly typed. The runtime already
tracks shape, axis labels, device, frozen status, and (for models)
a structural spec tree. None of that is exposed as a *type*. The
practical consequence: the canonical ML bug -- applying `softmax`
twice and feeding the result to `cross_entropy` -- is silently a
NaN factory rather than a type error. Less famously: `:describe`
is shape-and-values only, the trace JSON is shape-and-values only,
and an LLM agent has no schema to reason against beyond "every
binding is an `f64` array, hopefully of the right shape".

The runtime already does the work a type system would do; what is
missing is *naming*, *surfacing*, and *teaching*. Optional typing
in MLPL is a pedagogy + correctness saga, not a performance saga.

## What "type" means in MLPL

Five orthogonal axes. Optional typing chooses which to commit to.

| Axis        | What it tracks                                  | Today                       | Future                              |
|-------------|-------------------------------------------------|-----------------------------|-------------------------------------|
| Dtype       | f32 / f16 / bf16 / i32 / bool                   | `f64` only                  | deferred (performance-driven)       |
| Structural  | rank + dim sizes + (optionally) axis labels     | `LabeledShape` (Saga 11.5)  | promoted to type (Saga 26)          |
| Semantic    | Logit / Probability / Loss / Weight / etc.      | nothing                     | Saga 23 tags                        |
| Algebraic   | sum/product variants (`Distribution = ...`)     | implicit in `Value` enum    | Saga 24 distributions               |
| Effectful   | on-device, on-tape, frozen, requires-grad       | scattered; not unified      | unified under Saga 23 tag surface   |

Dtype stays last forever. The ranking puts performance last;
educational-first does not benefit from f16 vs f64. Quantization
is a separate research saga that can borrow the typing machinery
when it ships.

## What "optional" means

Three additivity rules govern every saga in this design:

1. **Untyped programs continue to run unchanged.** Any program
   that compiled and ran on v0.18.0 must compile and run on the
   typed-MLPL release that follows.
2. **Mixed programs are normal.** A typed binding can be passed
   to an untyped op, and an untyped binding can be passed to a
   typed op. Type predicates fire only when both sides agree they
   exist.
3. **Annotations are encouragement, not requirement.** A student
   should *want* to write `logits : Logit[batch, vocab]` because
   doing so is the lesson; they should not be *forced* to.

This is the gradual-typing model used by TypeScript, mypy, and
Sorbet: the runtime is the source of truth, and types are
additional predicates checked when present.

## Design priorities, ranked

The project-level goal ranking translates to typing-specific
priorities as follows.

### Educational (highest)

- **Type names are ML names.** `Logit`, `Probability`,
  `AttentionMap`, `Loss`, `Gradient` -- not `T<f32, [N, V]>` or
  `Numeric<Phantom=Unnormalized>`. Names *are* the curriculum.
- **Annotations are encouraged, not auto-inferred away.** Writing
  `softmax : Logit -> Probability` is the lesson; silent
  inference removes the lesson.
- **Type errors teach the underlying ML concept.**
  `cross_entropy: expected Logit, got Probability -- did you
  apply softmax twice? cross_entropy expects raw scores; if you
  already have probabilities, use nll(probs, targets).` Every
  type error is a tutor.
- **Trace and `:describe` make the type transformation visible.**
  Every traced event renders as
  `softmax: Logit[batch=4, vocab=8] -> Probability[batch=4, vocab=8]`.
  The trace stops being a debug aid and becomes a microscope on
  what each op *does*.

### Correctness

- The runtime is the source of truth. A type predicate that fires
  on entry to an op catches the bug before the wrong values
  propagate.
- The compile-to-Rust path lifts what it can to lower time, but
  this is a follow-up gain, not the goal.

### Utility

- Annotation cost stays bounded. Inference fills the gaps so
  unannotated code still benefits from auto-tagged outputs.
- Untyped corners stay first-class -- research code that doesn't
  yet know its types must be writable.

### Practicality

- No Hindley-Milner, no row polymorphism, no dependent types.
  Predicate-style runtime tags + light syntactic sugar is the
  ceiling.
- Annotation syntax extends Saga 11.5's `x : [batch, time, dim]`
  rather than introducing a new form.

### Maintainability

- *One* tag mechanism covers semantic + algebraic + effectful.
  Not three parallel systems.
- Tag definitions live in one module so the curated vocabulary is
  visible at a glance.

### Extensibility

- Users define their own tags (`tag(x, "MyMemoryRow")`) so
  research-grade concepts (Skill, Tool, Episode) don't gate on a
  language version bump.

### Performance (lowest)

- Every op may tag-check on entry. Every assignment may verify the
  predicate. Trace events carry the full type. None of this needs
  to be free.

## The six sagas

Each saga lands on its own milestone-* doc.

| # | Saga                                       | Goals served                       | Doc                                    |
|---|--------------------------------------------|------------------------------------|----------------------------------------|
| 23 | Typed ML values + typed traces            | educational, correctness           | `docs/milestone-typed-values.md`       |
| 24 | First-class Distributions                  | educational, correctness           | `docs/milestone-distributions.md`      |
| 25 | Inspectable ComputationGraph               | educational, correctness           | `docs/milestone-compute-graph.md`      |
| 26 | Annotation syntax + tutoring errors        | educational                        | `docs/milestone-typed-annotations.md`  |
| 27 | Typed Layer roles + walked `:describe mdl` | educational, maintainability       | `docs/milestone-typed-layers.md`       |
| 28 | User-defined tags                          | extensibility                      | `docs/milestone-user-tags.md`          |

The deferred Saga ?? -- static checks on the compile-to-Rust seam
-- is intentionally omitted. It is a correctness/performance gain
that ships when the dynamic story is solid; it does not move the
educational needle.

## The Tier A vocabulary

Saga 23 ships these tags as the curated core. Every later saga
extends from this list.

| Tag                | Auto-tagged by                          | Consumed (predicate-checked) by         |
|--------------------|-----------------------------------------|-----------------------------------------|
| `Logit`            | model `apply`, `linear` final           | `cross_entropy`, `sample`, `top_k`      |
| `Probability`      | `softmax`, `sigmoid`                    | `nll`, `kl_divergence`, `entropy`       |
| `LogProbability`   | `log_softmax`                           | `nll` accepts; `cross_entropy` rejects  |
| `Loss`             | `cross_entropy`, `mse`, `kl_divergence` | `grad`, `adam`, `momentum_sgd`          |
| `Gradient { wrt }` | `grad(loss, w)`                         | `:describe`, optimizer step             |
| `Weight`           | `linear`, `embed`, `attention`          | `:describe`, `freeze`/`unfreeze`        |
| `Bias`             | `linear`, `embed`                       | `:describe`, init/decay rules           |
| `Activation`       | `tanh_layer`, `relu_layer`, `softmax_layer` | `:describe`, `:hidden mdl k`        |
| `LearningRate`     | `cosine_schedule`, `linear_warmup`      | `adam`, `momentum_sgd`                  |
| `Labels`           | `one_hot` source side, dataset prep     | `confusion_matrix`, `cross_entropy`     |
| `AttentionMap`     | `attention_weights`                     | `svg(_, "heatmap")`, `:describe`        |

Distributions (Saga 24) and Layer roles (Saga 27) extend this
list.

## Example pedagogy

What a typed REPL session looks like once Sagas 23-26 ship:

```mlpl
mlpl> logits : Logit[batch, vocab] = apply(mdl, X)
mlpl> :describe logits
logits -- Logit[batch=4, vocab=8]
  domain: unnormalized scores
  next:   softmax -> Probability, log_softmax -> LogProbability
  values: -1.234 0.456 ...

mlpl> probs = softmax(logits, "vocab")
mlpl> :describe probs
probs -- Probability[batch=4, vocab=8]
  derived from: logits (via softmax over axis=vocab)
  invariant:    sum over vocab == 1.0 per row (verified)

mlpl> loss = cross_entropy(probs, Y)
type error in cross_entropy:
  expected: Logit[batch, vocab]
  got:      Probability[batch=4, vocab=8]
  hint:     cross_entropy expects raw scores, not probabilities.
            you applied softmax already (line 2). try one of:
              cross_entropy(logits, Y)   -- pass the original Logit
              nll(probs, Y)              -- negative log-likelihood
                                            from probabilities
```

Every line is unblocked by the typing layer. None of it is
unblocked by the language as it stands today.

## Open questions

- **Tag propagation through arithmetic.** `Logit + Logit` is
  another `Logit`. `Logit + Probability` is nonsense. `Loss + Loss`
  is `Loss`. The propagation table belongs in Saga 23 step 002.
- **Anonymous intermediates.** Side-table tags only attach to
  named bindings. `cross_entropy(softmax(logits), Y)` -- does the
  inner softmax result get a transient tag? Saga 23 ships
  side-table only; Saga 26 may revisit if the anonymous case
  proves load-bearing.
- **Numerical invariants.** Should the runtime *verify* that a
  `Probability`'s rows sum to 1.0? Yes for `:describe` (educational
  -- the printout shows the invariant). No for op entry (would
  re-run a sum on every call). Saga 23 step 005 nails the
  policy.
- **Typed traces and the eventual web viewer.** The trace JSON
  needs a stable schema for typed events. Saga 23 ships the
  schema; the deferred web-viz saga consumes it.
- **Compile-to-Rust handoff.** The `mlpl-lower-rs` static label
  check (compile-to-Rust saga) is the natural home for *static*
  type predicates. A future saga can lift Saga 23-26 checks to
  lower time without disturbing the dynamic story.

## See also

- `docs/typing.md` -- current untyped surface; this doc supersedes
  the "future directions" section there.
- `docs/typed-ml-concepts.md` -- the Tier A/B/C source proposal.
  Rolled into Sagas 23, 27, 28 of this plan.
- `docs/milestone-named-axes.md` -- Saga 11.5; structural-axis
  prerequisite for the annotation syntax in Saga 26.
- `docs/are-we-driven-yet.md` -- the agent-usability audit that
  motivates the educational ranking.
- `docs/research3.txt` -- the May 2026 research brief calling for
  first-class ML data types.
