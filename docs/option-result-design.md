# Option / Result / Error in sw-MLPL -- design note

Status: design discussion (2026-07-26). The Result layer is
SHIPPED and validated; the Option layer is a Stage 6 (nested
arrays) story. Companion: `docs/functional-lenses.md` (safe lens
gets are the first consumer).

## What is already in the language (validated by live eval)

```text
r = ok(42); is_ok(r)                      # 1
unwrap(ok(42))                            # 42
err_message(err("boom"))                  # boom
unwrap_or(err("boom"), 7)                 # 7
unwrap_or(ok([1,2,3]), 0)                 # 1 2 3
unwrap(err("idx 9 out of range"))         # hard error: UnwrapOnErr
is_ok(5)                                  # hard error: NotAResult (array is not a Result)
e = err({kind: "index", message: "axis 3 out of range"})
err_message(e)                            # {kind: index, message: axis 3 out of range}
rec = {kind: "index", msg: "oob"}; rec.msg  # oob
if is_ok(ok(1)) { 100 } else { 200 }      # 100
```

So: `Value::Result { ok, payload }` is a TAGGED SUM (Rust-shaped
`Result<T, E>`), constructed by `ok(v)` / `err(e)`, consumed by
`is_ok` / `is_err` / `unwrap` / `unwrap_or` / `err_message`. The
`err` payload is ANY value -- including a record -- so the
requested "Error object containing a String" exists today as a
convention: `err({kind: ..., message: ...})`.

## The question: Result as a pair of Options?

Proposal sketched in review: `Result(Some(Error), None)` vs
`Result(None, Some(v))` vs `Result(None, None)` -- a two-slot
product `(error_slot, value_slot)`.

Rust-flavored analysis: the product has four states and only two
are legal. `(Some, Some)` and `(None, None)` are the bugs-waiting
states; a tagged sum (`ok`/`err`, what we shipped) makes them
unrepresentable. If a third state is genuinely needed (pending /
not-yet-evaluated), NAME it rather than encoding it as
double-absence.

APL2-flavored analysis -- and this flips the picture: APL2 has no
sum types at all. Its idioms are:

- **Absence = the empty array** (zilde), not a None tag.
  Presence is tally 1, absence is tally 0, and `tally` is the
  discriminant. No new type; the array machinery IS the Option.
- **Result = the `(rc value)` pair** -- the classic APL/system
  convention. MLPL's `ok`/`err` is semantically exactly this
  pair, just opaque.
- **Illegal states are not forbidden, they are ABSORBED**: APL2
  fill/prototype semantics mean `first(empty)` yields a fill
  element instead of a crash. In that reading `(None, None)` is
  not illegal -- it degrades to fills. That is a coherent design,
  but it trades "errors surface loudly" for "errors flow through
  silently", and a REPL that teaches ML wants loud.

## Recommendation

1. **Keep the shipped `ok`/`err` sum as THE Result.** It already
   has the monadic accessors, and the loud failure modes
   (`UnwrapOnErr`, `NotAResult`) are what a teaching REPL wants.
2. **Standardize the Error object as a record payload**:
   `err({kind: "index", message: "axis 3 out of range", axis: 3,
   rank: 2})`. Canonical fields: `kind` (short machine tag),
   `message` (human string), plus free context fields. Safe
   wrappers (`u:get_safe`, future `put` builtin) build this
   shape. Optional later sugar: `error(kind, message)`.
3. **Option = the empty-or-one-element array (the APL2 answer).**
   `none = []` (works today: `tally([]) = 0`); `some(x) = [x]`
   for scalars today, `enclose(x)` for anything once Stage 6
   nested arrays land. Derived forms, all existing builtins:
   - `is_some(o)` = `gt(tally(o), 0)`
   - `unwrap_or(o, d)` = `first(concat(o, [d]))`
   No new value kind; `tally` is the tag. This slots into the
   Stage 6 scoping in `docs/apl2-staging-plan.md`.
4. **Bulk missing data stays array-native**: a validity MASK over
   the whole array (+ fill values), not per-cell Options. That is
   the deepest APL2 flavor and the one ML pipelines actually use
   (`batch_mask` already exists).
5. If a pending/empty THIRD state is ever needed, use the
   zilde-Result: `[]` where a Result is expected (tally 0 = "no
   result yet"), keeping the Result itself two-state.

## The projection-accessor synthesis (review round 2)

Second sketch from review: `result.getError() -> None|Some(reason)`
and `result.getValue() -> None|Some(value)`. This is the right
resolution of the pair-of-Options idea: the Result STAYS a tagged
sum internally, and the two Options become DERIVED projections --
Rust's `.ok()` / `.err()` exactly. Because they are computed from
one tag, the projections are complementary by construction: the
`(Some, Some)` and `(None, None)` illegal states cannot arise.

With zilde-Options the projections are plain arrays:

- `get_value(r)` -> `[]` when Err, `[v]`/`enclose(v)` when Ok
- `get_error(r)` -> `[]` when Ok, `[e]`/`enclose(e)` when Err

and immediately compose with existing primitives:
`first(concat(get_value(r), [default]))` IS `unwrap_or`. Surface
style: free functions (`get_value(r)`), not method syntax --
matching every existing builtin; `rec.field` stays reserved for
record field access. Scalar payloads work today; arbitrary
payloads want Stage 6 `enclose`.

## The three forces: type safety vs FP vs APL2

The general resolution is not to pick a winner but to assign each
force a LAYER, with pure functions as the seam:

1. **Control plane -- type safety wins.** The evaluator's value
   kinds are tagged sums (Rust enums); illegal states are
   unrepresentable; failures are loud (`UnwrapOnErr`,
   `NotAResult`). A teaching REPL must fail loudly at the
   boundary, so fills never silently swallow a mistake in
   control flow.
2. **Data plane -- APL2 wins.** Inside bulk array computation
   there are no tags: absence is emptiness (zilde), validity is a
   MASK (`eq`/`gt` already return 0/1 arrays, `batch_mask`
   exists), edge cases are absorbed by fills, and primitives stay
   rank-polymorphic so everything composes with everything. No
   per-cell Options -- a million tiny tags would destroy both the
   performance model and the notation.
3. **The seam -- FP wins.** Crossing between planes is done by
   pure total projections with algebraic laws: `ok`/`err` into
   the control plane; `get_value`/`get_error` (0-or-1 element
   arrays) back into the data plane; lenses as get/put pairs with
   the lens laws; error objects as plain records. No mutation, no
   partiality at the seam.

Rule of thumb when forces collide: ask "is this control flow or
data flow?" Control gets sums and loud errors; data gets masks
and fills; whatever crosses gets a pure projection. Existing
precedents in the codebase already follow this line: `EvalError`
is a Rust enum (control), comparison builtins return masks
(data), `unwrap_or` is a total seam function, the no-broadcasting
decision favors APL2 explicitness in the data plane, and the lens
`put` is an FP setter expressed in data-plane clothing.

## Prior art: how the APL family handles this

Survey of the derivatives (review round 3). The family consensus
is striking: NO Option/Result types in the data plane anywhere;
errors are loud exceptions until explicitly trapped; the trap is
a HIGHER-ORDER COMBINATOR (the seam is a function, not a type);
and the tagged rc-pair exists only as an opt-in projection.

- **APL2 / Dyalog APL**: errors signal and unwind; trapping via
  `Quad-EA`/`Quad-EC` (execute alternate), `:Trap`, `Quad-TRAP`,
  `Quad-SIGNAL`. Modern Dyalog exposes the error as `Quad-DMX` -- a
  namespace with `EN`/`EM`/`Message` fields, i.e. a STRUCTURED
  ERROR OBJECT, precedent for our `err({kind, message})` record.
  Absence: empty arrays + fill/prototype rules. Functional
  update: selective assignment `(sel x) <- v` and the `@` operator
  (`f@sel RIGHT-TACK x`) -- apply-at-selection, our masked lens put.
- **J**: errors abort; the adverse conjunction `u :: v` runs `u`
  and falls back to `v` on failure (`".::0:` = "parse or 0" --
  unwrap_or as a combinator); `try./catch.` in explicit code;
  `13!:11`/`13!:12` expose last error number/message. Absence:
  empties + fill (`!.f` fit customizes fills); a boxed empty is
  idiomatic "nothing". Functional update: amend `m}` .
- **K / q (kdb+)**: signal with `'`; protected evaluation
  `@[f;x;g]` / `.[f;args;g]` calls handler `g` WITH THE ERROR
  STRING; and the `.[f;args;:]` form returns a tagged pair --
  `(1b;result)` or `(0b;errmsg)` -- the rc-pair Result as an
  OPT-IN projection, exactly the get_value/get_error synthesis.
  Absence is the boldest data-plane design in the family: TYPED
  SENTINEL NULLS (`0N`, `0n`, backtick-null, per-type) that live
  inside ordinary vectors and propagate through primitives --
  per-cell missingness with zero per-cell tags. This is the
  industrial validation of "masks/sentinels, not Options, for
  bulk data".
- **BQN**: `F CATCH G` (glyph: circle-jot-diaeresis, BQN Catch) (Catch) traps, `system CurrentError` inspects, `!`
  asserts with a message; `F CATCH {default}` is unwrap_or as a
  combinator. No Option; "Nothing" (the middle-dot glyph) exists but is syntactic.
  BQN is also the family's FP high-water mark (first-class
  functions, lexical closures) -- and its `UNDER` (Under) is the
  family's fully general LENS: `F UNDER G` gets a view through `G`,
  applies `F`, and puts the result back through `G`'s structural
  inverse. Under = get/transform/put with the put-back derived
  automatically. The planned `put` builtin and any future
  `under`-style operator should cite this lineage
  (selective assignment -> J `}` -> Dyalog `@` -> BQN `UNDER`).
- **Nial**: the outlier -- FAULT values (`?err`) are data that
  propagate through array operations; errors-as-data-plane
  -sentinels, closest relative of "errors as values".
- **Uiua** (newer stack-array): `TRY` (star-diaeresis glyph) modifier + assert --
  same trap-combinator shape.

Consequences for sw-MLPL:

1. The layered proposal matches family practice: loud errors +
   opt-in projection (k's `.[;;:]`), structured error object
   (Dyalog `Quad-DMX`), sentinel/mask absence in bulk data (q
   nulls), combinator seam (J `::`, BQN `CATCH`).
2. A trap COMBINATOR (`attempt(u:f, u:handler)` or `f :: default`
   sugar) is the family's ergonomic core -- but it depends on
   first-class `u:` function values, reinforcing that staging
   item's priority.
3. BQN's Under is the north star for the lens work: `put` is the
   stepping stone; a structural-inverse `under` operator is the
   destination.

## Control-flow adapters: `?` and try/catch/finally (review round 4)

Requested: Rust-like `?` and Java-like try/catch/finally. The
two-plane model says these are not competing features -- they are
the TWO BRIDGES over the same seam, in opposite directions:

- **`catch` DEMOTES control-plane to data-plane**: a hard
  `EvalError` becomes an ordinary value the program can inspect.
- **`?` PROMOTES data-plane to control-plane**: an `err(...)`
  value becomes an early return that unwinds the enclosing
  function.

### try/catch (block form) -- implementable without first-class fns

```text
r = try { take(M, 0, 9) } catch e { fill([5], 0) }
```

- The whole form is an EXPRESSION (yields the body's value, or
  the handler's) -- functional flavor, unlike Java's
  statement-oriented try. MLPL blocks already yield their last
  expression, so this is consistent.
- `catch` catches HARD EvalErrors only, binding `e` to the
  canonical error record `{kind, message}` (the Dyalog `Quad-DMX`
  move). `err(...)` values are ordinary data and flow through
  untouched -- they are already "caught" by construction.
- `finally`: DEFERRED. Its job is resource cleanup and MLPL has
  no user-visible resources (no handles; load/save are atomic
  builtins). Once the block machinery exists, `finally { }` is
  cheap to add (run after body-or-handler, discard value,
  re-raise any pending error) -- add it when there is a resource
  to clean up, not before.
- Needs: parser block form + an eval intercept that converts
  `Err(EvalError)` from the body into the handler path. No
  first-class functions required -- same machinery class as
  `train N { }` / `device("x") { }`.

### `?` propagation -- reuses the existing `return` machinery

Inside a `def u:f(...) { }` body, `expr?` means: evaluate; if
`ok(v)` continue with `v`; if `err(e)` early-return that whole
Result from `u:f` (the evaluator already has a Return signal for
`return`). At top level (no enclosing function) it behaves as
`unwrap` -- loud, which is what a teaching REPL wants.

```text
def u:pipeline(s) {
  a = u:parse(s)?;
  b = u:validate(a)?;
  ok(u:emit(b))
}
```

Spelling: postfix `?` is available (ASCII-first; only conflicts
with APL roll/deal if a Unicode-APL compatibility layer ever
wants `?`, which can then live as `roll`). If parser cost delays
postfix syntax, an interim spelled form `check(expr)` (an eval
intercept like `ok`/`err`) has identical semantics -- retracing
Rust's own `try!` -> `?` history.

### The APL2-flavored end state: the trap combinator

Once `u:` functions are first-class values, add the family's
native form (J `::`, BQN `CATCH`, q `@[f;x;g]`):

```text
attempt(u:risky, u:handler)      # or sugar: u:risky :: u:handler
```

try/catch blocks are the ergonomic near-term surface; the adverse
combinator is the point-free final form. Both compile to the same
demote-to-value semantics.

### Error-handling demo (queued after the Life demo)

Placement (user direction 2026-07-26): category BASICS, placed
near Structure Zoo / Game of Life -- error handling is
foundational, not an advanced topic.

Four acts: (1) the two planes -- an `err(...)` value vs a hard
error, side by side with `disp`; (2) the railway -- safe lens
get returning `err({kind, message})`, `is_ok` branching,
`unwrap_or`, `get_value`/`get_error` projections; (3) ERRORS IN
USER-DEFINED FUNCTIONS (user direction) -- the guard-then-ok/err
shape as THE way a `u:` function reports failure (`u:get_safe`
is the template: validate inputs first, return
`err({kind, ...})` instead of letting a builtin explode mid-
body), a caller branching on the Result, and a two-stage `u:`
pipeline propagating the first failure; (4) the bridges --
`try/catch` demoting an out-of-bounds `take` into a handled
fill value, and `?` collapsing act 3's manual propagation into
one line per stage. Glossary: [[Result]], [[Error Handling]],
railway-oriented programming.

## First consumers

- Deep-lens demo (`u:get_safe`): out-of-bounds get returns
  `err({kind: "index", ...})`; demo contrasts the hard error from
  raw `take` with `unwrap_or(u:get_safe(...), fill_value)`.
- Future `put` builtin returns `err({kind: "shape", ...})` on
  slice-shape mismatch.
- `:upload` already binds `Ok({pixels,h,w})` / `Err("cancelled")`
  -- migrate its Err to the canonical record shape when touched.
