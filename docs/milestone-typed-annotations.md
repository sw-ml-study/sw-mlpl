# Annotation Syntax + Tutoring Errors Milestone (Saga 26)

## Why this exists

Saga 23 wires up the side-table tag mechanism and auto-tags the
common producers / consumers. But the *student writes nothing*.
Auto-tagging is silent inference: the tag appears in `:describe`
without the student having to declare it. Under the
educational-first ranking that's a problem -- writing
`logits : Logit[batch, vocab] = apply(mdl, X)` is the lesson, and
silent inference removes the lesson.

Saga 26 ships the annotation syntax that lets students *declare*
the tags they expect. The parser learns to accept type names
(not just axis labels) in the existing Saga 11.5 colon-annotation
form. The evaluator verifies the declared tag against the actual
tag at assignment time. And every annotation mismatch raises a
tutoring error -- not just "tag mismatch", but a 3-5 line tutor
message naming the most likely cause and concrete fixes.

This is the keystone *educational* saga of the optional-typing
rollout. Saga 23 makes typing possible; Saga 26 makes typing
*pedagogical*.

Goal ranking applied:

- **Educational** is the only ranked goal that justifies this
  saga. Without educational primacy, auto-tagging from Saga 23
  would be sufficient; annotation syntax adds nothing for
  correctness alone.
- **Practicality** rules the syntax: the annotation form must
  extend Saga 11.5's existing `x : [batch, time, dim]`
  annotation, not introduce a new form.
- **Maintainability**: tutoring messages live in one module
  (`crates/mlpl-eval/src/type_errors.rs` from Saga 23 step 007),
  not scattered across consumer ops.

## Non-goals

- Type *inference* in the static / functional sense. The
  evaluator continues to drive types from runtime values; the
  annotation is a *predicate* checked on assignment, not a
  source of truth.
- Generics / type variables. `Tensor<T>` -style abstraction is
  out of scope; the curated tag vocabulary is the surface.
- User-defined tag types in annotation position. Saga 28 ships
  user-defined tags as the `tag(x, "MyTag")` form; annotation
  syntax stays restricted to the curated vocabulary until the
  vocabulary stabilizes.
- Static checking on the `mlpl!` / `mlpl build` seam. A separate
  deferred saga lifts annotation predicates to lower time.
- Function-typed annotations. There are no first-class user
  functions in MLPL today; annotating function types is a
  follow-up.

## Quality requirements (every step)

Identical to Saga 23.

## What already exists

- Saga 11.5's `x : [batch, time, dim] = ...` annotation syntax in
  the parser. Saga 26 extends it; it does not replace it.
- Saga 23's `ValueTag` enum + side table + auto-tagging from
  producers + predicate-checking from consumers.
- `EvalError::TypeMismatch { op, expected, actual, hint }`
  variant + `crates/mlpl-eval/src/type_errors.rs` tutoring hint
  module (Saga 23 step 007). Saga 26 extends the hint module
  with assignment-site hints.
- `:describe` rendering of typed values (Saga 23 step 005).

## Phases

### Phase 1: Parser surface

Extend the Saga 11.5 annotation form to accept type names before
the axis-label list:

```mlpl
logits : Logit[batch, vocab] = apply(mdl, X)
loss : Loss = cross_entropy(logits, Y)
W : Weight[in, out] = param[2, 8]
```

Grammar additions:

- `Annotation := TypeName? AxisLabelList?` (currently
  `Annotation := AxisLabelList`).
- `TypeName` is a curated identifier set (the Saga 23 Tier A
  vocabulary).
- An annotation with only a type name (no axis labels) is
  legal: `loss : Loss = cross_entropy(...)`.
- An annotation with only axis labels (the Saga 11.5 form)
  remains legal, unchanged.
- Reject unknown type names at parse time with a tutoring
  message ("Logit, Probability, Loss, Gradient, Weight, Bias,
  Activation, LearningRate, Labels, AttentionMap are the
  curated tags; user-defined tags are a Saga 28 feature").

### Phase 2: Assignment-time predicate

The evaluator checks the annotation's tag (and labels) against
the right-hand side's actual tag at assignment.

- Tag matches actual tag: bind, propagate the (annotated) tag
  forward.
- Tag mismatches actual tag: raise
  `EvalError::TypeMismatch { op: "assignment", expected,
  actual, hint }` with a tutoring hint chosen from the
  assignment-site catalog (Phase 3).
- Tag annotated, value untagged: bind, attach the annotated
  tag to the side table (the annotation acts as a manual tag
  for an untagged producer).
- Tag unannotated, value tagged: bind unchanged (the auto-tag
  flows through; this is the Saga 23 default).

### Phase 3: Assignment-site tutoring messages

Extend `crates/mlpl-eval/src/type_errors.rs` with
assignment-site hints. Each entry maps `(expected_tag,
actual_tag)` to a 3-5 line tutor:

- `(Logit, Probability)` -- "you annotated `: Logit` but the
  right-hand side is a `Probability`. likely cause: you applied
  softmax already. fix: drop the annotation, or wrap the rhs in
  `log_softmax` to recover a `LogProbability`."
- `(Probability, Logit)` -- "you annotated `: Probability` but
  the right-hand side is a `Logit`. fix: wrap the rhs in
  `softmax(rhs, "axis=...")` to convert."
- `(Loss, Probability)` -- "you annotated `: Loss` but the
  right-hand side is a `Probability`. likely cause: you forgot
  the loss function. fix: pass through `cross_entropy(rhs, Y)`
  or `mse(rhs, Y)`."
- ... one entry per common Tier A pair. The catalog is
  maintained as a single Rust file so the curated set is
  reviewable in one diff.

### Phase 4: `:tag` and `:untag` REPL commands

Manual escape hatches for when the auto-tagger guesses wrong.

- `:tag <name> <kind>` -- attaches a tag to a binding.
- `:untag <name>` -- clears a tag.
- `:tags` -- lists all tagged bindings (already shipped in Saga
  23 step 005; this phase polishes the listing to include
  manual vs auto-tag origin).

### Phase 5: Annotated parameter declarations

`param[shape]` and `tensor[shape]` constructors learn to accept
a tag annotation in their assignment:

```mlpl
W : Weight[in=2, out=8] = param[2, 8]
b : Bias[1, 8] = param[1, 8]
```

Without an annotation, the auto-tagger applies the existing
Saga 23 rule (Weight for matrix-shaped params, Bias for
row-vector-shaped params adjacent to a Weight). With an
annotation, the user's declared tag wins; mismatch raises a
tutoring error.

### Phase 6: Annotated function arguments

A future user-defined-function feature is out of scope, but
*built-in* function calls grow the ability to mention their
expected types in the error message:

- `cross_entropy(logits : Logit[batch, vocab], y : Labels[batch,
  num_classes=vocab]) -> Loss` is the documented signature in
  the type-error hint, even if there are no per-arg annotations
  in the call site.
- `:describe cross_entropy` (the existing function-introspection
  command) prints the typed signature.

This phase is small in code but high in pedagogical leverage:
the student now sees typed signatures everywhere they look.

### Phase 7: Tutorial lesson + retrospective + release

- New web REPL lesson "Typing the Pipeline" placed after Saga
  23's "Typed ML Values" lesson. Walks: untyped baseline ->
  annotated pipeline -> deliberately-wrong annotation (tutor
  fires) -> annotated `param`/`tensor` -> reading a typed
  signature in `:describe`.
- Update `docs/using-typed-values.md` with an "Annotations"
  chapter.
- Update `docs/saga.md`, `docs/status.md`,
  `docs/are-we-driven-yet.md`.
- Bump REPL banners; rebuild `pages/`; tag the release.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | parser-type-annotations    | 1 | grammar additions + parse-time vocabulary check |
| 002 | assignment-tag-predicate   | 2 | runtime tag check on assignment |
| 003 | tutoring-assignment-hints  | 3 | extended `type_errors.rs` catalog |
| 004 | tag-untag-repl-commands    | 4 | `:tag` / `:untag` polish; auto-vs-manual origin in `:tags` |
| 005 | annotated-param-tensor     | 5 | `W : Weight[...] = param[...]` predicate |
| 006 | typed-signatures-describe  | 6 | builtin signatures shown in `:describe` and error hints |
| 007 | typed-annotations-tutorial | 7 | new web REPL lesson |
| 008 | typed-annotations-release  | 7 | docs, banners, pages rebuild, release tag |

Eight steps.

## Success criteria

- `logits : Probability = apply(mdl, X)` raises a tutoring error
  whose hint quotes "you annotated `: Probability` but the
  right-hand side is a `Logit`. fix: ..."
- `W : Weight[in=2, out=8] = param[2, 8]` binds, and `:describe
  W` shows `Weight[in=2, out=8]` with origin "annotated".
- `:describe cross_entropy` prints
  `cross_entropy(logits : Logit, y : Labels) -> Loss`.
- The "Typing the Pipeline" lesson runs in the browser without
  external network access.
- Every Saga 6-22 demo continues to run, and every annotated
  Saga 23 demo continues to run with its annotations honored.
- Unknown tag in annotation (`x : Foo = 1`) raises a parse-time
  error pointing at the curated vocabulary list.
- Quality gates green; pages deployed; release tagged.

## Risks and open questions

- **Annotation scope.** Saga 11.5's annotation attaches to the
  *binding name*, not to anonymous intermediates. Saga 26
  inherits that. `cross_entropy(softmax(logits) :
  Probability, Y)` -- can we annotate an anonymous intermediate?
  Default: no. The annotation form is reserved for
  assignment statements. Revisit if the educational case
  emerges.
- **Vocabulary stability.** Saga 28 (user-defined tags) might
  later promote a user tag to the curated vocabulary; the
  parser's curated-set check needs to be data-driven so the
  vocabulary is not hardcoded in three places.
- **Backwards compatibility.** Saga 11.5's
  `x : [batch, time, dim]` form must still parse. The grammar
  addition is strictly extending the annotation prefix; verify
  with parity tests against every annotated Saga 11.5+ demo.
- **Compile-to-Rust path.** The `mlpl!` macro and `mlpl build`
  CLI need to handle the new annotation form even if they
  cannot yet *check* it. Lower-time checking is deferred; the
  parsers must agree.
- **Error message clutter.** A tutoring hint of 3-5 lines per
  error can drown out the rest of the REPL output if many
  errors fire in a tutorial. Mitigation: hints are emitted
  once per binding per session by default, suppressed on
  repeat with a one-line "(see earlier hint for ...)".
