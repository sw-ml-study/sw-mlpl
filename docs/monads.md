# Monads and sw-MLPL: an analysis

Written 2026-08-05 (user question during the mlplunit-unblock
saga). This is a design/analysis document -- it references
planned work and belongs to the planning docs, not the
user-facing set.

## What exists today: one monad, implemented concretely

sw-MLPL ships the ERROR MONAD as a concrete value kind rather
than an abstraction:

- `Value::Result { ok, payload }` is Either-shaped: `ok(x)` /
  `err(e)` construct it, so `ok` plays the role of `pure` /
  `return`.
- The postfix `?` is the error-monad bind specialized to the
  identity continuation: unwrap on `ok`, EARLY-RETURN the whole
  `err` otherwise. This is exactly Rust's model -- propagation
  as a syntax form -- not Haskell's, where `>>=` is a first-class
  operator you can pass around and abstract over.
- Eliminators: `is_ok` / `is_err`, `unwrap` (hard on err),
  `unwrap_or` (Haskell's `fromMaybe`), `err_message`, and the
  structured error record for caught hard errors
  (docs/error-handling.md: "two planes, two bridges").

## The second, very APL encoding: Maybe as data shape

`get_value(r)` and `get_error(r)` project a Result onto a
0-or-1-element vector (zilde when absent). Because array
operations propagate emptiness, a pipeline can early-exit BY
SHAPE: an empty vector flows through arithmetic, `compress`,
and reductions as emptiness, and `tally(x)` is `is_some`. This
is the list monad restricted to length <= 1, encoded in data
rather than control -- the most array-language answer to "other
ways to express early return besides `?`":

- `?` = control-plane early return (statement-shaped, one
  value).
- Zilde projection = data-plane propagation (pipeline-shaped,
  and it VECTORIZES: an `is_ok` mask plus `compress` is batch
  error handling over many fallible results at once, which
  Haskell expresses as `catMaybes`/`rights`).

Both are shipped; the zilde idiom is under-demonstrated and
deserves demo coverage on its own merits.

## What cannot be expressed yet, and the precise blocker

Monadic COMBINATORS -- `lift`/`fmap` (`map_ok`), `and_then`
(bind with a real continuation), `or_else` -- are higher-order:
they take a function. User-defined functions are not yet
first-class values, so there is nothing to pass. That is the
whole blocker, and it is already scheduled twice over: contract
item 3 of the mlplunit-unblock program (`:u:name` references +
`call(f, args...)`) and the APL2 higher-order saga's
prerequisite (`each` / `scan` / `outer` wait on the same
capability).

Once callables land, the combinators are small builtins:

```text
map_ok(:u:f, r)      # fmap: apply f inside ok, pass err through
and_then(:u:f, r)    # bind: f returns a Result itself
or_else(:u:f, r)     # recover: f sees the error payload
```

That is Rust-level monad support: one blessed monad with real
composition. Full Haskell-level support -- user-DEFINED monads,
do-notation, abstraction over the monad itself -- requires a
static type system (typeclasses / traits) that sw-MLPL
deliberately does not have; encoding it dynamically would buy
generality nobody has asked for at the cost of the language's
teachability. Non-goal.

## The related pattern that IS wanted: bracket

mlplunit's fixture lifecycle (contract item 6) needs
guaranteed-finally: setup, use, teardown-even-on-error. In
monadic terms that is `bracket` -- the resource-safe companion
of the error monad. It cannot be faked by host-side source
appending (mlplunit's doc says so correctly) and it will need a
language/runtime mechanism regardless of how the combinator
surface evolves. When it is designed, it should compose with
`?`: an early return through a bracketed region still runs the
teardown.

## Recommendation

1. Now: nothing blocking; `?` + `unwrap_or` + zilde projections
   cover today's programs.
2. With the callables step (mlplunit P1): add `map_ok` /
   `and_then` / `or_else` as builtins in the same saga -- they
   are the first real exercise of `call()` and make fallible
   pipelines compose without nesting.
3. With the fixture-lifecycle item (P2): design
   bracket/guaranteed-finally as a language construct that
   respects `?`.
4. Demo debt worth paying regardless: a "Result pipelines" demo
   showing the control-plane (`?`) and data-plane (zilde /
   masks) side by side.
