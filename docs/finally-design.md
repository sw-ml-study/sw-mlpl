# Guaranteed-finally / bracket: design

Status: APPROVED as recommended (user, 2026-08-06) and
implemented (bracket-core + bracket-errors steps).
Upstream contract: mlplunit's sw-MLPL-changes-needed.md item 6
(fixture and suite lifecycle), whose eight required behaviors
need a language/runtime guaranteed-finally mechanism -- their
doc states plainly that host-side source appending cannot
provide the guarantees. docs/monads.md recommendation 3 is the
same item seen from the error-monad side: bracket is the
resource-safe companion of the error monad, and it must compose
with `?`.

## The three candidate surfaces

### A. `try { ... } finally { ... }` statement

Extend the existing `try { } catch e { }` form with a `finally`
block that runs on every exit path.

- Pro: familiar; textual scoping.
- Con: the guarantee must survive EVERY unwind path -- `?`
  early-return, hard evaluator errors, `break`/`continue`,
  nested try -- which means the evaluator grows a genuine
  unwinding protocol (pending-finally stack consulted by every
  propagation site). That is a cross-cutting change to the
  interpreter's control flow, and each `?` site pays for it.
- Con: a statement form gives the runner no VALUE-level way to
  retain both diagnostics when use and teardown both fail
  (behavior 6): the finally block has no principled channel to
  merge its failure with the in-flight one.

### B. `bracket(setup, use, teardown)` builtin (RECOMMENDED)

Three function references (the machinery `call` / the
combinators already use); the guarantee lives INSIDE one
builtin, so no interpreter-wide unwinding protocol exists at
all.

- Pro: smallest correct surface. The evaluator already knows
  how to catch hard errors at one boundary (`try`/`catch` does
  exactly this); bracket reuses that capture at the `use` call
  and the `teardown` call, and sequencing inside the builtin IS
  the guarantee.
- Pro: value-shaped results -- both-fail diagnostics merge into
  the returned error record naturally (behavior 6).
- Pro: composes outward: `bracket(...)?` early-returns AFTER
  teardown has run, by construction. A suite runner (mlplunit's
  or an MLPL-written `u:suite`) is a loop over
  `bracket(reg.before_each, t, reg.after_each)`.
- Con: not a general `finally` for arbitrary non-fixture code.
  Accepted: the contract's need is fixture lifecycle; a general
  statement form remains addable later (A is not foreclosed by
  B, and B would become its obvious implementation core).

### C. A full `suite({...})` builtin

Implement mlplunit's demonstrated record directly.

- Con: bakes RUNNER policy (ordering, reporting, skip handling,
  parameterization) into the evaluator; their contract
  explicitly wants parameterization/lifecycle as library
  surface, not special-cased evaluator behavior. Rejected --
  the fixture updates to the accepted public syntax per their
  own doc.

## Recommended semantics: `bracket(setup, use, teardown)`

All three arguments are function references (`:u:name`;
builtin references are rejected -- lifecycle hooks are user
code). Contract behaviors map as numbered:

1. `setup()` is called with no arguments and produces the
   fixture value (any Value kind).
2. If `setup` returns `ok(x)`, the fixture is `x`; a plain
   value `v` is treated as `ok(v)` (hooks stay writable
   without Result ceremony). If `setup` returns `err(e)` or
   hard-errors, `use` and `teardown` are both SKIPPED
   (behavior 5) and bracket returns `err(...)` -- the setup
   error payload, or the structured `{kind, message}` record
   for a hard error.
3. `use(fixture)` is called once. Hard evaluator errors are
   CAUGHT at this boundary exactly as `try`/`catch` catches
   them, becoming `err({kind, message})`, so teardown always
   has its chance (behavior 4).
4. `teardown(fixture)` runs exactly once whenever setup
   succeeded -- after a pass, a returned `err`, or a caught
   hard error (behaviors 3 and 4). It receives the SAME
   fixture value `use` received (behavior 2's isolation is the
   caller's loop; bracket never reuses a fixture because it
   only ever has one).
5. Result precedence (behavior 6):
   - use ok + teardown ok -> use's result (ok or plain value,
     as returned).
   - use failed + teardown ok -> use's failure, unchanged.
   - use ok + teardown failed -> `err(teardown's failure)` --
     a leaked resource is a real failure.
   - both failed -> use's failure stays PRIMARY; the teardown
     diagnostic is RETAINED by attaching it to the error
     payload: if the primary payload is a record it gains a
     `teardown_error` field, otherwise it becomes
     `{message: <primary>, teardown_error: <teardown's>}`.
6. `bracket(...)` is an expression; `bracket(...)?` composes
   -- early return happens after teardown by construction.
7. `before_all`/`after_all` (behavior 7) need nothing new:
   they are one more bracket wrapped around the per-test loop.
8. File/process isolation (behavior 8) is unaffected -- the
   runner may still fork per test; bracket works identically
   inside that process.

Errors bracket itself raises (not Results): wrong arity, a
non-reference argument, a builtin reference -- all structured
eval errors naming `bracket` and the offending argument.

## What stays out

- No general `try`/`finally` statement in this step (see A).
- No enforcement of `timeout_ms`, no event stream, no process
  controls -- contract item 7 owns those.
- No suite runner: mlplunit composes one from `tests()` /
  `test_info` / `call` / `bracket`; an MLPL-native demo suite
  is a natural follow-up demo, not evaluator work.

## Implementation steps (after review)

- bracket-core -- the builtin over three `:u:` references:
  setup/use/teardown sequencing, plain-value-as-ok, setup-skip
  rule. TDD.
- bracket-errors -- hard-error capture at use and teardown,
  precedence and `teardown_error` merging, `?` composition,
  structured misuse errors. TDD; run mlplunit
  check-capabilities expecting fixture-lifecycle's
  guaranteed-finally prerequisite satisfied (their gate also
  lists test-registry, already shipped).
- bracket-docs -- lang-reference, glossary, usage, catalog
  rows, wiki matrix; monads.md recommendation 3 marked
  shipped.

## Open questions for review

1. Naming: `bracket` (Haskell lineage, used throughout the
   analysis docs) vs `with_fixture` (self-describing) vs
   `ensure`. Proposed: `bracket` -- the docs already teach it
   and the concept is search-friendly.
2. Should `use` failures that were HARD errors be
   distinguishable from returned `err`s in bracket's result?
   Proposed: yes for free -- the caught form is the structured
   `{kind, message}` record `try`/`catch` already produces, so
   the payload shape carries the distinction.
3. Zero-fixture usage: allow `bracket(:u:setup, :u:use,
   :u:teardown)` where `use`/`teardown` take no parameter
   (fixture ignored)? Proposed: no special case -- hooks
   declare one parameter; arity errors already name the
   referent.
