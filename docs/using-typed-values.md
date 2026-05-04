# Using Typed ML Values (Saga 23 retrospective)

**Status:** shipped 2026-05-02 as v0.19.0. Optional typing
keystone. Educational-first ranking (educational > correctness
> utility > practicality > maintainability > extensibility >
performance).

For the design context, see
`docs/optional-typing-design.md`. For the planning sketch see
`docs/typed-ml-concepts.md` (now superseded).

## What shipped

The curated Tier A typed-value vocabulary as a side-table tag
on `Environment`, with auto-tagging, predicate-checked
consumers, propagation, typed `:describe` / `:vars` / `:tags`
/ `:untag`, typed trace JSON events, and the
`HINT_*` tutoring catalog.

Programs that did not adopt typed values keep working
unchanged (gradual-typing additivity rule). Programs that
do adopt them get the canonical double-softmax bug as a
pre-train-step type error with a copy-pasteable fix.

## The vocabulary

Eleven curated tags in `mlpl_core::ValueTag`:

| Tag                         | Carries           | Auto-tagged by                                      |
|-----------------------------|-------------------|-----------------------------------------------------|
| `Logit`                     | -                 | `apply(model, X)` when model tail is `Linear`/`LinearLora` |
| `Probability`               | -                 | `softmax`, `sigmoid`, `apply` ending in `softmax_layer` |
| `LogProbability`            | -                 | (reserved; `log_softmax` not yet a builtin)         |
| `Loss { kind }`             | `LossKind`        | `cross_entropy` (others reserved for Saga 24)       |
| `Gradient { wrt }`          | param name        | `grad(loss, wrt_ident)`                             |
| `Weight { layer, name }`    | layer + slot      | `linear`, `embed`, `attention` constructors         |
| `Bias { layer }`            | layer             | `linear`                                            |
| `Activation { layer, kind }`| layer + `ActivationKind` | `apply` ending in `tanh_layer`/`relu_layer`  |
| `LearningRate`              | -                 | `cosine_schedule`, `linear_warmup`                  |
| `Labels { num_classes }`    | class count       | (reserved; `one_hot` source side not auto-tagged yet) |
| `AttentionMap`              | -                 | `attention_weights`                                 |

`LossKind` covers `CrossEntropy`, `Mse`, `KlDivergence`,
`Custom`. `ActivationKind` covers `Tanh`, `Relu`, `Sigmoid`,
`Softmax`. Both are siblings in `mlpl-core`; the typing layer
deliberately keeps a separate copy of `ActKind` so `mlpl-core`
stays independent of the model DSL.

## Producer auto-tagging

Two layers of auto-tagging fire at the assignment site:

1. `auto_tag::for_assign` runs `from_fncall` for direct
   FnCall right-hand sides (`probs = softmax(L, 1)` produces
   Probability) and `apply_tag` for model `apply` calls
   (`out = apply(mdl, X)` walks the structural tail of `mdl`
   to derive Logit / Probability / Activation).
2. Param creation in `model_dispatch::eval_linear` /
   `eval_embedding` / `eval_attention` tags the `__linear_W_N`
   / `__embed_E_N` / `__attn_W{q,k,v,o}_N` param names with
   Weight / Bias as it allocates them.

Producer tags ALWAYS take precedence over propagation tags.
If `for_assign` returns `Some(t)`, that's the tag; propagation
runs only when the producer dispatcher returns `None`.

## Predicate consumers

`crates/mlpl-eval/src/type_errors.rs` ships two predicate
helpers:

- `check_logit_consumer` runs at the call site of
  `cross_entropy`, `sample`, `top_k`. Untagged args pass.
  Logit args pass. Probability / LogProbability / (for
  cross_entropy only) Labels are rejected with a four-part
  tutoring hint:
  1) what the op expected and why,
  2) what it got,
  3) likely cause,
  4) one or two concrete fixes in copy-pasteable MLPL.
- `check_loss_consumer` runs at `adam` / `momentum_sgd`.
  Untagged or Loss-tagged first args pass. Probability /
  Logit / etc. raise TypeMismatch with a hint pointing at
  the missing loss function.

The full hint catalog is documented in
`contracts/eval-contract/tutoring-hints.md`.

## Tag propagation

`crates/mlpl-eval/src/tag_propagate.rs` defines the rules
(documented in `contracts/eval-contract/tag-propagation.md`):

- Same-family arithmetic (`Logit + Logit`, `Loss + Loss`):
  keep the family.
- Tagged + Untagged: tagged side wins.
- Domain-mixing arithmetic (`Logit + Probability`,
  `Loss + Logit`): TypeMismatch with `HINT_DOMAIN_MISMATCH`
  pointing at softmax / log / cross_entropy / mse as the
  bridge.
- `transpose` / `reshape_labeled` / `label` / `relabel`:
  preserve.
- `reshape`: clear (shape reflow loses semantic identity).
- Reductions (`mean`, `reduce_add`, `reduce_mul`, `argmax`):
  Loss survives; Probability / Logit / etc. clear.
- Unary negation: preserve.
- Bare-identifier alias: copy.

## REPL surface

Five typed REPL commands shipped:

- `:describe <name>` -- typed header + per-tag body. For
  Probability bindings, the body verifies the per-row sum
  invariant (1e-5 tolerance) and reports verified or violated.
- `:vars` -- adds a tag column at the end of each row.
- `:tags` -- new -- lists every tagged binding sorted by
  name.
- `:untag <name>` -- new -- clears a tag deliberately.
- `:tag` -- not shipped in Saga 23; manual tag attachment
  syntax lands with the annotation work in Saga 26.

Typed `:describe` example:

```
mlpl> probs = softmax(L, 1)
mlpl> :describe probs
probs -- Probability
  shape: [2, 3]
  values: 0.3329 0.3340 0.3331 ...
  row-sum invariant: verified (max deviation 1.11e-16)
```

## Typed trace events

`mlpl_trace::TraceEvent` gains `input_types` /
`output_type` fields, both
`#[serde(default, skip_serializing_if = ...)]`. Untagged
programs serialize byte-identically; tagged programs add
the metadata.

For a `loss = cross_entropy(L, Y)` line where `L` is
Logit-tagged, the trace JSON gains:

```json
{
  "op": "assign",
  "input_types": ["Logit", null],
  "output_type": { "Loss": { "kind": "CrossEntropy" } }
}
```

Skip-when-all-none policy: the input_types vec is omitted
when no input carries a tag, keeping output-only-tagged
events terse.

The full schema is documented in
`contracts/eval-contract/typed-trace.md`.

## The canonical bug, finally caught

Before Saga 23:

```
loss = cross_entropy(softmax(logits, 1), Y)
# silent NaN factory; only diagnosed by reading
# loss values mid-training and noticing they're nonsense
```

After Saga 23:

```
mlpl> loss = cross_entropy(softmax(logits, 1), Y)
type mismatch in cross_entropy: expected Logit, got Probability
  hint: this consumer expects Logit (raw scores) because
        cross_entropy / sample / top_k assume an
        unnormalized score per class. you got Probability.
        likely cause: you applied softmax already. fix:
        drop the softmax and pass the original Logit --
        `loss = cross_entropy(logits, y)` -- or, once
        Saga 24 ships nll, `loss = nll(probs, y)`.
        double-softmax inside cross_entropy is the
        canonical NaN factory at scale.
```

## Walking the lesson

The web REPL ships a "Typed ML Values" lesson
(`apps/mlpl-web/src/lessons_advanced.rs::TYPED_ML_VALUES`)
that walks a student through producer auto-tagging, the
double-softmax bug, propagation, and `:tags` / `:untag`. The
35 example lines double as a worked example of the typing
layer; nine smoke tests in
`crates/mlpl-eval/tests/typed_values_lesson_smoke.rs` keep
the lesson invariants from drifting.

## What's NOT shipped (deferred)

These items appear in `docs/optional-typing-design.md` but
land in later sagas:

- **Annotation syntax** (`logits : Logit[batch, vocab] = ...`)
  -- Saga 26. The Saga 23 layer is auto-tagging only; the
  user does not yet *write* tags.
- **Inline-FnCall predicate checking inside builtin args**
  -- the `cross_entropy(L + P, Y)` case where the binop is
  inline does not raise the domain-mismatch error in
  Saga 23. The infer walk swallows propagation errors so
  predicate consumers see the binop as untagged.
- **Distribution tags** -- Saga 24 ships
  `Categorical`/`Gaussian`/`Mixture` as a separate enum
  variant; predicate rules grow then.
- **LayerRole tags** -- Saga 27 promotes ModelSpec variants
  to typed roles for `:describe mdl` walking.
- **User-defined tags** (`define_tag("MemoryRow", ...)`) --
  Saga 28.
- **Static checks on the `mlpl!` / `mlpl build` lower path**
  -- a follow-up saga lifts the dynamic predicates to lower
  time. Educational-first ranking puts this last; the
  dynamic story takes precedence.
- **`log_softmax`, `mse`, `kl_divergence`, `nll` builtins**
  -- forward-looking tags reference these but the producer
  builtins are not yet at the language surface. Saga 24
  ships them as part of the Distribution work.
- **Web-REPL trace panel** -- the trace JSON schema carries
  typed events but no UI surfaces them yet. Future trace
  viewer work consumes the schema.
- **`:tag <name> <kind>` manual attachment** -- the inverse
  of `:untag`. Lands with annotation syntax in Saga 26.

## Out of scope (intentionally)

- Dtype (f32 / f16 / bf16 / i32). Performance is ranked last;
  educational-first does not benefit from finer numeric
  control. Quantization will revisit.
- Hindley-Milner inference, generics, traits. The runtime
  side-table predicate model is the practical ceiling.
- Type system localization. Hints are English; future i18n
  can wrap the catalog.

## See also

- `docs/optional-typing-design.md` -- umbrella design.
- `docs/milestone-typed-values.md` -- this saga's milestone
  doc.
- `contracts/eval-contract/tag-propagation.md` -- the
  propagation table in prose.
- `contracts/eval-contract/typed-trace.md` -- the trace
  schema contract.
- `contracts/eval-contract/tutoring-hints.md` -- the hint
  catalog with copy-pasteable fix vocabulary.
- `apps/mlpl-web/src/lessons_advanced.rs::TYPED_ML_VALUES`
  -- the web REPL lesson.
