# Typed ML Values + Typed Traces Milestone (Saga 23)

## Why this exists

The keystone of the optional-typing rollout sketched in
`docs/optional-typing-design.md`. Every later typing saga (24
distributions, 25 compute-graph, 26 annotations, 27 layer roles,
28 user-defined tags) consumes the side-table tag mechanism that
Saga 23 ships.

Concretely, this saga makes the canonical ML bug -- applying
`softmax` twice and passing the result to `cross_entropy` -- a
type error caught at the call site, with a tutoring message
instead of a NaN factory. It also makes `:describe` and the trace
JSON pedagogically useful for the first time: every traced event
renders as `softmax: Logit[batch=4, vocab=8] ->
Probability[batch=4, vocab=8]` so a student can read a trace and
learn what each op *does*.

Goal ranking applied (educational > correctness > ... >
performance):

- **Educational** is served by typed traces, typed `:describe`,
  and tutoring error messages -- the *visibility* of the type
  transformation is the lesson, not just the type check.
- **Correctness** is served by the runtime predicates fired on op
  entry.
- **Performance** is explicitly not a goal: every op may tag-check
  on entry, every trace event may carry the full type, every
  `:describe` may re-walk the metadata.

## Non-goals

- Annotation syntax. `x : Logit[batch, vocab] = ...` lands in
  Saga 26 once the side-table machinery is solid.
- Dtype. Everything stays `f64`. f16/bf16/i32 are deferred
  forever-or-until-quantization.
- Anonymous-intermediate tags. Tags attach to named bindings only.
  Saga 26 may revisit if the anonymous case proves load-bearing.
- Static checking on the `mlpl!` / `mlpl build` seam. Lifting tag
  predicates to lower time is a separate, deferred saga.
- User-defined tags. Curated Tier A vocabulary only; user tags
  ship in Saga 28.
- Distributions, computation graphs, layer roles. Sagas 24, 25,
  27 respectively.

## Quality requirements (every step)

Per CLAUDE.md:

1. TDD: failing test first, then implementation, then refactor.
2. Quality gates green before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` (ASCII-only) if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.
6. `.agentrail/` files committed alongside source changes.

## What already exists

- `Value::Array` / `Value::Model` / `Value::Str` enum in
  `mlpl-eval`. Tags would live on a new `Environment` side table
  keyed by binding name.
- `Environment::frozen_params: HashSet<String>` (Saga 15) -- the
  precedent for a name-keyed side table on `Environment`.
- `Environment::experiment_log` (Saga 12) -- precedent for
  serializable per-binding metadata.
- `:vars` / `:describe` REPL commands (Saga 11) ready to consume
  tag metadata in their formatters.
- Trace JSON export (Saga 4) ready to grow a typed-event schema.
- The Tier A vocabulary table in `docs/optional-typing-design.md`
  is the source of truth for which tags ship in Saga 23.

## Phases

### Phase 1: Tag enum + side table

- New `ValueTag` enum in `mlpl-eval` (or `mlpl-core` if reused by
  trace export) with the Tier A variants from
  `docs/optional-typing-design.md`. Variants carry the metadata
  the design doc lists (e.g. `Gradient { wrt: String }`,
  `Weight { layer: String, name: String }`).
- New `Environment::tags: HashMap<String, ValueTag>` side table.
  Tags follow re-binding by default (per-name, not per-value).
- `set_tag(name, tag)` / `get_tag(name)` helpers internal to the
  evaluator. No language surface yet.
- Zero behavior change in any existing demo.

### Phase 2: Auto-tagging from producer ops

Wire the ops that *produce* typed values to tag their results.

- `softmax` / `sigmoid` -> `Probability`.
- `log_softmax` -> `LogProbability`.
- `linear` / `embed` / `attention` constructors -> `Weight` /
  `Bias` for the params they create.
- `tanh_layer` / `relu_layer` / `softmax_layer` apply outputs ->
  `Activation { layer, kind }`.
- `cross_entropy` / `mse` / `kl_divergence` -> `Loss { kind }`.
- `grad(loss, w)` -> `Gradient { wrt: w }`.
- `cosine_schedule` / `linear_warmup` scalars -> `LearningRate`.
- `attention_weights` -> `AttentionMap`.
- Model `apply` final-layer output -> `Logit` (heuristic: the
  outermost chain's last `Linear` produces logits unless followed
  by softmax).

### Phase 3: Predicate-checked consumers

Wire the ops that *consume* typed values to assert predicates on
entry. Every assertion failure raises a new
`EvalError::TypeMismatch { op, expected, actual, hint }` variant
with a tutoring `hint` string.

- `cross_entropy(logits, y)` -- expected `Logit`; reject
  `Probability` with hint "you applied softmax already; use
  `nll(probs, y)` or pass the original Logit".
- `nll(probs, y)` -- expected `Probability` or `LogProbability`.
- `sample(logits, t, seed)` / `top_k(logits, k)` -- expected
  `Logit`.
- `entropy(probs)` -- expected `Probability`.
- `confusion_matrix(preds, labels)` -- expected `Labels` for
  argument 2 with `num_classes` matching argument 1's last dim.
- `adam(loss, ...)` / `momentum_sgd(loss, ...)` -- expected
  `Loss` for argument 1.

Untyped values pass every predicate (gradual-typing additivity
rule 1).

### Phase 4: Tag propagation

Define and implement the tag propagation table for every op that
takes typed inputs.

- `Logit + Logit -> Logit`, `Loss + Loss -> Loss`.
- `Logit + Probability` -- structurally fine but semantically
  nonsense -- error with hint "Logit and Probability live in
  different domains; combine them via cross_entropy or convert
  one with softmax / log".
- Elementwise ops between a tagged and untagged operand pass the
  tag through.
- `transpose`, `reshape_labeled` preserve tags.
- `reshape` clears tags (semantic identity is lost on shape
  reflow).
- Reductions (`reduce_add`, `mean`) over a tagged value yield a
  scalar with the same tag *only* if the tag survives reduction
  (`Loss` does, `Probability` does not -- a partial sum of probs
  is not a Probability).

This phase ships the propagation table as code plus a contract
doc under `contracts/eval-contract/tag-propagation.md`.

### Phase 5: Typed `:describe`

Rewrite `:describe <name>` to consume tags.

- Header line shows the tag: `logits -- Logit[batch=4, vocab=8]`.
- Body lines tailored per tag:
  - `Probability` shows the per-row sum invariant (verified
    once at print time, with a "verified" / "violated" note).
  - `Gradient` shows `wrt` plus the last optimizer step it was
    consumed by, if known.
  - `Weight` shows layer, init seed, and grad-computed-at-step
    if the env has a record.
  - `Activation` shows the producing layer and the activation
    kind.
- `:vars` learns to show the tag in its one-line summary.
- `:tags` -- new REPL command listing every tagged binding with
  its tag, sorted by tag kind.
- `:untag <name>` -- escape hatch for clearing a wrong auto-tag
  (uncommon but necessary).

### Phase 6: Typed trace events

Extend the trace JSON schema to carry tags.

- Each `TraceEvent` gains an optional `input_types` /
  `output_types` field (arrays of tag names with their
  metadata).
- Trace formatter prints
  `softmax: Logit[batch=4, vocab=8] -> Probability[batch=4,
  vocab=8]` instead of the current shape-only line.
- A new `contracts/eval-contract/typed-trace.md` pins the
  schema so the deferred web-viz saga has a stable target.
- The CLI `--trace-json out.json` output is the canonical
  artifact; the web REPL trace panel learns to render typed
  events.

### Phase 7: Tutoring error messages

Rewrite every `EvalError::TypeMismatch` site to emit a tutoring
hint, not just the type names. Hints are stored as constants in a
new `crates/mlpl-eval/src/type_errors.rs` module so they are
discoverable in one place. Each hint is two-to-five lines:

1. What the op expected and why.
2. What it got.
3. The most likely cause.
4. One or two concrete fixes (with example MLPL syntax).

This phase is small in code but high in pedagogical leverage; it
is the load-bearing educational deliverable of Saga 23.

### Phase 8: Tutorial lesson + retrospective + release

- New web REPL tutorial lesson "Typed ML Values" placed
  immediately after "Named Axes" and before "Model Composition".
  Walks: untyped baseline -> auto-tagged softmax/cross_entropy
  pipeline -> deliberately-wrong double-softmax (shows the
  tutoring error) -> tag propagation through arithmetic ->
  `:describe` and `:tags` tour -> typed trace export.
- New `docs/using-typed-values.md` retrospective + user guide.
- Update `docs/typing.md` to redirect "future directions" at this
  saga and the rest of the optional-typing rollout.
- Update `docs/saga.md` and `docs/status.md`.
- Update `docs/are-we-driven-yet.md` -- move Tier A items to HAVE.
- Bump REPL banners.
- Rebuild `pages/`.
- Tag the release.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | tag-enum-side-table         | 1 | `ValueTag` enum + `Environment::tags` table; no behavior change |
| 002 | auto-tag-producers          | 2 | softmax/sigmoid/log_softmax/grad/linear/etc. auto-tag outputs |
| 003 | predicate-consumers         | 3 | cross_entropy/nll/sample/top_k/adam check tag preconditions |
| 004 | tag-propagation             | 4 | propagation table + contract doc |
| 005 | typed-describe-vars-tags    | 5 | `:describe`/`:vars`/`:tags`/`:untag` consume metadata |
| 006 | typed-trace-events          | 6 | trace JSON schema + CLI/REPL formatter changes |
| 007 | tutoring-error-messages     | 7 | rewritten `TypeMismatch` hints in `type_errors.rs` |
| 008 | typed-values-tutorial       | 8 | new web REPL lesson |
| 009 | typed-values-release        | 8 | docs, banners, page rebuild, release tag |

Nine steps. Steps 002 and 003 may merge if the producer/consumer
pairs cluster cleanly (e.g. softmax+cross_entropy as one TDD
slice).

## Success criteria

- `cross_entropy(softmax(logits), Y)` raises a structured
  `TypeMismatch` with the tutoring hint quoted in
  `docs/optional-typing-design.md` (or close to it).
- `:describe logits` after `apply(mdl, X)` shows
  `Logit[batch=N, vocab=V]` without a manual `:tag` call.
- `:describe probs` after `softmax(logits, "vocab")` shows
  `Probability[batch=N, vocab=V]` plus a row-sum invariant note.
- The trace JSON for `softmax(logits, "vocab")` carries the
  full input/output type pair; `--trace-json` produces a file
  that round-trips through a deserializer with all type metadata
  intact.
- Every existing Saga 6-22 demo runs unchanged. No demo regresses.
- `docs/are-we-driven-yet.md` shows the Tier A typing rows moving
  from PLAN/CONS to HAVE.
- Quality gates green; pages deployed; release tagged.

## Risks and open questions

- **Re-binding semantics.** `W = W - lr * grad(...)`. Does the
  tag carry through? Side-table answer: tags are per-name and
  survive reassignment unless the right-hand side is auto-tagged
  with a different tag. Test this in step 001.
- **Logit detection at the model boundary.** The "outermost
  chain's last Linear produces logits" heuristic is right for
  every Saga 13 demo but wrong for a model that ends in a
  softmax-or-sigmoid layer. Mitigation: walk the spec tree at
  `apply` time and decide based on the structural tail. Document
  the rule in the contract.
- **Probability invariant verification cost.** Verifying the
  row-sum invariant on every `:describe` is fine; doing so on
  every op entry is not. Policy: verify in `:describe` only,
  document the policy in the contract, surface a `verify(x)`
  built-in for users who want an explicit check.
- **Tag propagation through `map(f, x)`.** `map` is a
  rank-polymorphic escape hatch; the tag should pass through
  unless `f` is structurally domain-changing (e.g. `exp`, `log`).
  Default to "pass through unless the function is on a curated
  domain-changing list".
- **Anonymous intermediates.** `cross_entropy(softmax(logits),
  Y)` -- the inner softmax has no binding to attach a tag to.
  Saga 23 ships side-table only and lets the inner op fall back
  to the structural rule (the consumer checks the *expression
  shape*, not the binding tag, when no binding exists).
  Whether this is enough for a clean error is the biggest open
  question; revisit in Saga 26 if it isn't.
