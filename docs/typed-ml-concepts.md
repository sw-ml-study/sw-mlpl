# Typed ML Concepts (design proposal)

**Status:** design proposal, not yet a saga. Captures the
"typing assistance for ML concepts" direction the user asked
about on 2026-04-09 after the v0.7.0 release. Intended to seed
a future Saga (call it 11.7 or 12.5 depending on dependency
ordering).

For the current typing model, see `docs/typing.md`. For the
inserted Saga 11.5 that adds labeled axes (a prerequisite for
some of this), see `docs/milestone-named-axes.md`.

## Why types for ML concepts?

MLPL today represents the entire ML world as `DenseArray` plus
the model DSL's `ModelSpec`. That is enough to *run* a model,
but it is not enough to *talk about* one. The user (human or
agent) cannot ask:

- "What kind of layer is this?"
- "Is this tensor a weight, a gradient, an input, or an
  activation?"
- "Which layers in this chain produce hidden states I might want
  to inspect?"
- "Which parameters belong to the encoder vs the decoder?"
- "Is this number a learning rate or a temperature?"

Answers to these questions exist *in the user's head* and *in
naming conventions* but not in the language. Adding semantic
types to ML concepts would let the language carry that meaning,
expose it via `:describe`, validate it at op boundaries, and
give agents a structured surface to reason against.

This document is a sketch of what such a type system could look
like, what concepts deserve to be typed first, and what stages
of adoption are feasible without rewriting the language.

## What "type" means in this proposal

Three different things, layered:

1. **Tags** -- a runtime label attached to a `DenseArray` that
   says "this is a weight" or "this is a hidden activation".
   Carried in the value, used by `:describe` and by error
   messages, validated only when an op explicitly asks. Cheap.
2. **Roles** -- a structured tag with extra fields. A `Weight`
   role might carry its layer name, its position in the layer
   (e.g. `W` vs `b`), and its initializer seed. A
   `LearningRate` role might carry its schedule.
3. **Interfaces** -- a contract on what a value supports. A
   `Layer` interface promises `apply(layer, X)` works and
   advertises its expected input and output shapes. An
   `Optimizer` interface promises a `step(params, grad)` style
   call.

Tags are the cheapest to add and the most user-visible. Roles
build on tags. Interfaces are the largest jump because they
require dispatch on the role at the call site.

This proposal recommends starting with **tags**, opportunistically
upgrading hot ones to **roles** as concrete demos require it,
and deferring **interfaces** until a future saga (likely
post-Saga 11.5 named axes, since interfaces want shape labels
to be useful).

## ML concepts that deserve a type

In rough order of payoff per unit of language complexity.

### Tier A: ship first

These are the concepts where the lack of typing already hurts
demo readability and `:describe` output today.

| Concept            | Today                                  | Proposed type tag       | Why                                                                |
|--------------------|----------------------------------------|-------------------------|--------------------------------------------------------------------|
| **Weight**         | a `DenseArray` marked as a `param`     | `Weight { layer, name }` | distinguish trainable matrices from data tensors and from biases   |
| **Bias**           | a `DenseArray` marked as a `param`     | `Bias { layer }`         | same distinction; biases initialize differently and rarely decay   |
| **Input**          | a `DenseArray` you happen to pass to `apply` | `Input { kind }`     | enable shape and label checks against a layer's expected input     |
| **Output / logits** | a `DenseArray` returned by `apply`     | `Output { layer }`       | mark predictions for downstream loss/eval consumers                |
| **Hidden activation** | an intermediate `DenseArray` from `tanh_layer`/`relu_layer` | `Activation { layer, kind }` | makes "where is the hidden state from layer 3?" answerable |
| **Loss**           | a scalar `DenseArray` named "loss"     | `Loss { kind }`          | distinguish MSE / cross-entropy / KL for `:describe` and reporting |
| **Gradient**       | the result of `grad(loss, w)`          | `Gradient { wrt }`       | trace which param a gradient belongs to without naming convention  |
| **Learning rate**  | a scalar `DenseArray` named "lr"       | `LearningRate { schedule }` | catch mistakes like passing an array where a scalar lr is expected |
| **Label vector**   | an integer-valued `DenseArray`         | `Labels { num_classes }` | allow `confusion_matrix` to validate matching class counts         |

These are all *additive*: every concept already exists as a
`DenseArray`. The proposal is to attach a tag, not change the
storage. Programs that ignore the tags still work.

### Tier B: build once Tier A is in place

| Concept             | Today                                       | Proposed surface           |
|---------------------|---------------------------------------------|----------------------------|
| **Layer**           | a `ModelSpec` variant                       | a `Layer` role with `input_kind`, `output_kind`, and `params` |
| **Hidden layer**    | a non-final `Layer` in a `chain`           | a `Layer` with `is_hidden=true` (set automatically by `chain`) |
| **Activation layer**| `Activation(ActKind)` in `ModelSpec`        | first-class `ActivationLayer` role with no params              |
| **Optimizer**       | `momentum_sgd` / `adam` built-ins           | an `Optimizer` role carrying state across calls               |
| **Schedule**        | `cosine_schedule` / `linear_warmup` scalars | a `Schedule` role with a `step(t)` interface                  |
| **Dataset**         | tuples of `DenseArray` produced by `moons`  | a `Dataset` role with `inputs`, `labels`, `n_classes`, etc.   |
| **Trace**           | the `Trace` JSON                            | a typed `Trace` value queryable from MLPL                     |

These are mostly *upgrades* to things that already exist; the
goal is to attach explicit role metadata so the language can
surface it.

### Tier C: research-grade, plan but defer

These overlap with the paper-driven backlog
(`docs/paper-driven-development.txt`) and deserve their own
sagas before they get types.

- **Memory module** (paper #3 Engram, paper #13 MSA): tag rows
  in an external memory table with their freshness, source, and
  retrieval count.
- **Skill** (paper #10): a typed `Skill` value bundling code +
  prompt + tests + permissions.
- **Tool** (paper #4): a typed `Tool` value with a schema and an
  invocation channel.
- **Episode / rollout / reward** (paper #4, #7, #8): typed RL
  primitives with structured trajectories.
- **Plan / executor** (paper #8 PEARL): typed plan trees that
  can be inspected before execution.
- **Voices / debate / critic / judge** (paper #19): typed
  multi-perspective reasoning roles.

Each of these is ambitious enough that it should be designed as
its own subsystem first; the type tag is the *least* of the
work.

## What `:describe` should print once tags exist

Today (v0.7.0):

    :describe W1
    W1 -- array
      shape: [2, 8] (trainable param)
      values: -0.1234 0.4567 ...

Proposed (post-tags):

    :describe W1
    W1 -- Weight (layer: encoder.linear_1)
      shape: [in=2, out=8]
      role:  trainable parameter
      init:  randn(seed=11) * 0.5
      grad:  not yet computed
      values: -0.1234 0.4567 ...

`:describe` becomes the primary self-description surface for
typed values. Every tier-A tag carries enough metadata to
populate a few extra lines.

## Where the tags live

Two implementation paths, with different cost / benefit:

### Path 1: side table on Environment

Add `pub(crate) tags: HashMap<String, ValueTag>` to `Environment`.
A `set_tag(name, tag)` helper attaches a tag to a binding by
name. `:describe` reads the tag from the side table when
formatting.

- Cheap: no change to `DenseArray`, no change to `Value`,
  no parser changes.
- Limited: tags only apply to *named bindings*. Anonymous
  intermediates lose their tag immediately.
- Good first step. Lets us answer "what is `W1`?" without
  invasive changes.

### Path 2: typed wrapper around DenseArray

Add a new `Value::Typed { array: DenseArray, tag: ValueTag }`
variant. Every op that produces a result either propagates the
tag or strips it. Built-ins gain tag-aware versions.

- Expensive: every existing op needs to handle the new variant,
  every demo needs to check what gets stripped, the tag
  propagation rules are non-trivial (does `W1 + W2` produce a
  `Weight` or just an array?).
- Powerful: anonymous intermediates retain their identity. The
  loss `mean((apply(mdl, X) - Y) * (...))` could be tagged
  `Loss { kind: MSE }` automatically.
- Probably too much work for v0.7.x. Reserve for after Saga 11.5.

**Recommendation:** start with Path 1 (side table). It is
additive, contained to `mlpl-eval`, and unblocks the `:describe`
improvements that motivate this proposal. Path 2 is the sequel.

## Typing-assistance loop for users (and agents)

The point of all this is to enable a *conversation* between the
user and the workspace:

    mlpl> mdl = chain(linear(2, 8, 11), tanh_layer(), linear(8, 2, 12))
    mlpl> :describe mdl
    mdl -- chain(2 hidden layers, 1 output layer)
      input:  shape [n, 2]
      output: shape [n, 2]
      hidden activations: 1 (after linear_1, kind=tanh)
      params: 4 -- W1[2,8], b1[1,8], W2[8,2], b2[1,2]
    mlpl> :describe W1
    W1 -- Weight (layer: mdl.linear_0)
      shape: [in=2, out=8]
      role:  trainable parameter
      grad:  computed at last adam() call (step 47)
    mlpl> :hidden mdl 0     ; brand-new command
    [Activation tanh, layer mdl.linear_0]
    shape: [n, 8]
    last computed at step 47

Every line of that session is unblocked by tagging. None of it
is unblocked by the language as it stands today.

Agents benefit even more. A tagged value gives an LLM a
concrete schema to follow when constructing or repairing a
program: "I need a `Weight` of shape `[in, out]` to plug into
this `Linear` layer", instead of "I need an array, hopefully of
the right shape, hopefully not a bias, hopefully not a
gradient".

## Open questions

- **Tag propagation for anonymous values.** Tier-A side-table
  tags only attach to bindings. Do we want to add a
  "register on assign" rule so that `g = grad(loss, W1)` is
  automatically tagged `Gradient { wrt: W1 }`? Probably yes.
  How far does it go?
- **Subtyping.** Is a `Weight` also an `Array`? Yes -- ops that
  take an array should accept a tagged value transparently. The
  reverse is not true: a built-in that demands a `Weight` should
  refuse a plain array.
- **Mutation.** When `W1 = W1 - lr * grad(...)`, does the tag
  carry through the assignment? Yes if the rhs is "obviously" a
  weight update (Path 2 territory). With a side table, tags are
  per-name and survive reassignment by default.
- **User-defined tags.** Should users be able to define their
  own roles? `tag(x, "MyMemoryRow")`? Yes eventually, but
  probably not in the first step. Start with the curated Tier A
  vocabulary.
- **Interaction with named axes.** A `Weight` *and* labeled axes
  are complementary: the role says "this is a trainable weight
  matrix", the labels say "axis 0 is `in_features`, axis 1 is
  `out_features`". They compose without conflict.
- **Models as parameter trees.** `:describe mdl` should walk the
  ModelSpec and pretty-print each layer with its inferred input
  and output shapes. This is achievable today (even before
  tags) and could be a small follow-up.

## Suggested first-step deliverable

Something small and concrete to validate the approach before
committing to a full saga:

> Add a `ValueTag` enum with the Tier-A variants, a
> `tags: HashMap<String, ValueTag>` side table on `Environment`,
> a `:describe` rendering that consumes the tag, and a
> `:tag <name> <kind>` REPL command for manually attaching a
> tag. Wire `linear()` and friends to auto-tag the params they
> create as `Weight` / `Bias`, and `grad()` to auto-tag its
> result as `Gradient { wrt }`. No new built-ins beyond `:tag`,
> no parser changes, no new `Value` variant. One commit, one
> step, fully reversible.

If that lands cleanly and the `:describe` output proves useful
in practice, the next step is the same treatment for `Loss`,
`LearningRate`, `Labels`, and `Activation`. Then iterate.

## Saga shaping

This is at least one full saga's worth of work, possibly two:

- **Saga 11.7 -- Typed ML concepts (Tier A):** `ValueTag`
  enum, side table, `:tag` / `:untag` REPL commands,
  auto-tagging from `linear()` / `chain()` / `grad()` / `adam()`
  / `cosine_schedule()`, expanded `:describe` output for tagged
  values, tutorial lesson "Typed ML Concepts", regression
  tests. ~6-8 steps.
- **Saga 11.8 -- Typed ML concepts (Tier B):** `Layer` /
  `Optimizer` / `Schedule` / `Dataset` roles, layer-aware
  `:describe mdl` walker that prints the model as a typed tree,
  contract checks at op boundaries (e.g. "loss expression for
  `adam` must be a `Loss`-tagged scalar"). ~6-8 steps.

Tier C overlaps with the paper-driven backlog and should be
folded into the relevant feature sagas (memory in 18, skills
and tools in 18, etc.) rather than a single typing saga.

## See also

- `docs/typing.md` -- the *current* typing model (untyped
  homogeneous arrays).
- `docs/milestone-named-axes.md` -- Saga 11.5, which adds
  labeled axes (compatible and complementary).
- `docs/are-we-driven-yet.md` -- the audit that surfaced the
  gap between the agent-driven wishlist and what MLPL has today.
- `docs/paper-driven-development.txt` -- the paper backlog that
  most Tier C concepts will land alongside.
