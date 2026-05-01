# User-defined Tags Milestone (Saga 28)

## Why this exists

Sagas 23-27 ship a *curated* tag vocabulary: Logit, Probability,
Loss, Gradient, Weight, Bias, Activation, LearningRate, Labels,
AttentionMap, plus the layer / optimizer / schedule / dataset
roles. That vocabulary covers the well-understood ML primitives.

It does not cover the research-grade concepts on the
`docs/paper-driven-development.txt` backlog: Memory rows,
Skills, Tools, Episodes, Rewards, Plans, Voices, Critics,
Judges, etc. `docs/typed-ml-concepts.md` calls these "Tier C" and
explicitly defers them to per-feature sagas. Each of those
features eventually lands as its own saga and could ship its own
hardcoded tag, but doing so means a parser change every time a
new research feature wants typed values.

Saga 28 makes the typing surface *open*. Users (and future
sagas) can define their own tags via `tag(x, "MyTag")` and
`define_tag("MyTag", { ... })`. The runtime enforces the tag
predicate; `:describe` renders the tag; the trace JSON carries
it. The curated vocabulary stays as the documented
"first-citizen" set, but any program can extend it.

Goal ranking applied:

- **Extensibility** is the leading goal. The whole point of
  Saga 28 is to unblock research-feature sagas without
  parser changes.
- **Educational** is served by letting a tutorial author
  define a teaching-purpose tag (`SimpleAttention`,
  `RegularizationLoss`) without being constrained to the
  curated vocabulary.
- **Maintainability** is bounded: user-defined tags share the
  same machinery as the curated tags; there are no parallel
  systems.

## Non-goals

- Tag *inference* over user-defined tags. Auto-tagging is
  reserved for the curated producers/consumers; user tags are
  always manually attached via `tag(...)`.
- Tag inheritance / subtyping (`MyAttention extends Attention`).
  Out of scope; if useful later, ships in its own saga.
- Tags with executable predicates beyond shape / labels / value
  invariants. Out of scope -- predicates are declarative, not
  callbacks.
- Tag namespaces / modules. The user tag set is flat.
  Convention-based naming (`MemoryRow`, `Skill`) is the
  recommended discipline.
- Cross-session persistence of tag definitions. Tag definitions
  live in the `Environment`; they vanish when the REPL exits.
  A future "session save / restore" saga handles this.

## Quality requirements (every step)

Identical to Saga 23.

## What already exists

- Saga 23 `ValueTag` enum + side table + auto-tagging machinery.
- Saga 26 annotation syntax + assignment-time predicate checks.
- Saga 27 layer-role mechanism (precedent for declarative
  metadata attached to a value).
- The Saga 23 `:tag` / `:untag` REPL commands.

## Phases

### Phase 1: Tag definition surface

A new `define_tag` builtin and a parallel REPL command.

- `define_tag("MemoryRow", {
    requires_shape: [N=*, D=*],
    requires_invariants: ["row_norm <= 1.0"],
    description: "a row in an external memory table"
  })` registers a user tag with declarative predicates.
- `Environment::user_tags: HashMap<String, UserTagDef>` stores
  the registry.
- The `ValueTag` enum gains a `User { name: String }` variant
  whose validation walks the registry.
- `:define_tag <name> { ... }` REPL command for interactive
  registration.
- `:user_tags` lists every defined tag with its predicate.

### Phase 2: Tag attachment

- `tag(x, "MemoryRow")` attaches a user tag to a binding.
  Verifies the registry predicate at attachment time; rejects
  with a tutoring error if the predicate fails.
- `untag(x)` (already shipped in Saga 23 as `:untag`)
  generalizes to clear a user tag.
- Tag attachment via the Saga 26 annotation syntax:
  `m : MemoryRow[N, D] = randn(seed, [10, 8])`.

### Phase 3: Tag predicate language

Define the small declarative predicate vocabulary that
`define_tag` accepts.

- `requires_shape: [...]` -- shape predicate; `*` means any
  size, named dims must match.
- `requires_labels: ["batch", "..."]` -- label predicate.
- `requires_invariants: [...]` -- a small set of
  string-encoded invariants chosen from a curated list:
  - `"row_sum == 1.0"` (with tolerance)
  - `"row_norm <= 1.0"`
  - `"all_positive"`
  - `"all_in_range(min, max)"`
  - `"integer_valued"`
- `description: "..."` -- human-readable purpose, shown in
  `:describe`.

The vocabulary is curated and small. Arbitrary user-defined
predicates (closures) are out of scope.

### Phase 4: Trace and describe integration

- `:describe x` for a user-tagged value prints the tag name,
  the registered description, and the satisfied / unsatisfied
  invariants.
- The trace JSON schema (Saga 23 step 006) gains a
  `user_tag: { name, predicates }` slot for typed-trace
  events whose values carry user tags.
- `compute_graph(loss)` (Saga 25) walks past user-tagged
  intermediates without losing the tag in the graph value.

### Phase 5: Promotion path documentation

A user tag that proves widely useful can graduate to the
curated vocabulary. Document the promotion criteria in
`docs/optional-typing-design.md`:

- Used in at least three independent demos.
- Has a stable, agreed-upon predicate.
- Has a producer op that auto-tags (curated tags are
  auto-tagged; user tags are manual).

The promotion itself is a one-step Saga that adds the variant
to the curated `ValueTag` enum, wires up the auto-tagger, and
flags it as deprecated in the user-tag registry.

### Phase 6: Demo + tutorial + retrospective + release

- `demos/memory_row_typed.mlpl` -- a tiny external-memory
  retrieval demo using a user-defined `MemoryRow` tag with
  the `row_norm <= 1.0` invariant.
- New web REPL lesson "Defining your own tags" placed after
  Saga 27's "Typed Layer Tree".
- `docs/using-typed-values.md` gets a "User-defined tags"
  chapter.
- Update `docs/saga.md`, `docs/status.md`,
  `docs/are-we-driven-yet.md`.
- Bump REPL banners; rebuild `pages/`; tag the release.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | user-tag-registry        | 1 | `define_tag`, `Environment::user_tags`, `:user_tags` |
| 002 | user-tag-attachment      | 2 | `tag(x, "...")`, annotation-syntax acceptance |
| 003 | predicate-vocabulary     | 3 | curated invariant set + verification at attachment |
| 004 | user-tag-trace-describe  | 4 | trace schema + `:describe` rendering |
| 005 | promotion-path-doc       | 5 | criteria documented; one curated promotion as worked example |
| 006 | user-tag-demo-tutorial   | 6 | demo + new web REPL lesson |
| 007 | user-tags-release        | 6 | docs, banners, pages rebuild, release tag |

Seven steps.

## Success criteria

- `define_tag("MemoryRow", { requires_shape: [N=*, D=*],
  requires_invariants: ["row_norm <= 1.0"] })` registers
  successfully; `:user_tags` lists it.
- `m : MemoryRow[N=10, D=8] = ...` attaches the tag and
  verifies the row-norm invariant at assignment.
- A row with norm 1.5 raises a tutoring error pointing at
  the unsatisfied predicate.
- `:describe m` for a `MemoryRow`-tagged binding shows the tag
  name, registered description, and verified invariants.
- `demos/memory_row_typed.mlpl` runs end-to-end.
- All existing demos still pass; the curated tag vocabulary
  remains unchanged externally.
- Quality gates green; pages deployed; release tagged.

## Risks and open questions

- **Predicate vocabulary growth.** Every research feature might
  ask for one more invariant kind. Discipline: invariants ship
  in a single curated list; new invariants land as one-step
  follow-ups, not as user-supplied closures.
- **Naming conflicts with the curated vocabulary.** A user
  attempting to `define_tag("Logit", ...)` should be rejected
  with a hint pointing at the curated set. The reverse case --
  a future curated tag colliding with a popular user tag --
  triggers the promotion path.
- **Invariant verification cost.** `row_norm <= 1.0` on a
  million-row table is non-trivial. Verify on attachment only;
  do *not* re-verify on every op entry. Document the policy.
- **Cross-session persistence.** Tag definitions are
  per-session. A future session-save saga can persist the user
  tag registry; until then, programs that rely on user tags
  must re-define them at the top of every session (a one-line
  pattern; document it).
- **Educational tension.** User-defined tags blur the
  curriculum: a student following one tutorial sees `MemoryRow`,
  another sees `EngramSlot`, a third sees `MemoryCell`. Mitigate
  by curating a *recommended* set in
  `docs/using-typed-values.md` user-tag chapter and by
  promoting the strongest user tags to the curated vocabulary
  via the documented path.
