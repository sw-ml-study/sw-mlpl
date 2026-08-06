# First-class user-function references: design

Status: DRAFT for review (saga mlplunit-unblock, step 009).
Upstream contract: mlplunit's sw-MLPL-changes-needed.md item 3
(acceptance fixture `tests/capabilities/callable_function_case.mlpl`).
The language reference predicted this feature: "forward-compatible
with first-class functions: `:foo` can lift to a function value."

## Surface

```text
def u:double(x) { x * 2 }

f = :u:double              # a REFERENCE -- names it, does not run it
call(f, 21)                # 42: uniform invocation
suite = {double: :u:double, halve: :u:halve}
call(suite.double, 21)     # registries are records of references
```

- `:u:name` is the quoted form of a user function, completing the
  three-kinds-of-name story: `u:name(...)` calls, `:u:name`
  quotes. (Today `:u:double` silently lexes as TWO tokens --
  `:u` then `:double` -- a misparse nobody can be relying on;
  the lexer change claims it.)
- `call(f, args...)` invokes any reference value: a `:u:name`
  reference runs the user function; a builtin reference
  (`:disp`, `:add`) runs the builtin -- one calling model for
  both, which is the "documented common calling model" the
  contract asks for.
- References live anywhere a Value lives: variables, record
  fields, function arguments, return values. NUMERIC ARRAYS stay
  numeric (f64) -- a registry is a record, not an array; this is
  a documented boundary, not a gap.
- No closures: a reference IDENTIFIES a definition; user
  functions capture nothing today and the reference changes
  nothing about that. Re-defining `u:name` after taking a
  reference means the reference calls the NEW definition
  (late binding by name -- simple, and matches how the REPL's
  redefinition workflow already behaves). Documented explicitly.

## Value model

- New `Value::UserFnRef { name }` (kind string `user-fn-ref`),
  displayed `:u:name` -- distinct from `BuiltinRef` so registry
  diagnostics and `repr` stay honest about what they hold.
  `equal` compares by name; `repr` renders `:u:name`.
- `:describe` on either reference kind describes the referent
  (the user function's signature + doc-string, or the builtin's
  catalog row).

## Diagnostics (contract requirements)

- `call` with a non-reference first argument: a tutoring error
  naming the three reference forms.
- Arity mismatch on invocation: the existing user-fn arity error
  already names the function; `call` routes through the same
  path so the error identifies the REFERENCED function, not
  `call` itself.
- An unknown referent at call time (`:u:name` taken, definition
  never made or since cleared): structured error naming
  `u:name`.

## The monad-combinator rider (docs/monads.md recommendation)

Three small builtins land in the same saga step, as the first
real exercise of `call`:

| Builtin | Semantics |
|---|---|
| `map_ok(f, r)` | `ok(x)` -> `ok(call(f, x))`; `err` passes through untouched. |
| `and_then(f, r)` | `ok(x)` -> `call(f, x)` where `f` itself returns a Result; `err` passes through. |
| `or_else(f, r)` | `err(e)` -> `call(f, e)` (recovery); `ok` passes through. |

These give the error monad real composition (Rust-level support;
see docs/monads.md for why full Haskell-style monads are a
non-goal).

## Steps

- callables-ref -- lexer (`:u:name` single token) + parser +
  `Value::UserFnRef` + display/describe/equal/repr arms, TDD.
- callables-call -- `call(f, args...)` for both reference kinds,
  arity/identity diagnostics, record-registry tests mirroring
  the mlplunit fixture shape.
- callables-combinators -- map_ok / and_then / or_else + docs
  (usage, lang-reference three-kinds table update, glossary);
  run mlplunit check-capabilities expecting
  callable-user-functions to flip AVAILABLE.
- Then: metadata design (contract item 4) as its own pause.

## Open questions for review

1. Late binding by name (proposed) vs snapshotting the
   definition at reference time? Late binding is simpler, keeps
   redefinition workflows working, and matches the registry use
   case; snapshotting would need closure-like storage for no
   asked-for benefit.
2. `call` on a builtin reference with the wrong arity: builtins
   report their own arity errors today (naming the builtin) --
   good enough, or should `call` add a frame ("in call(:disp,
   ...)")? Proposed: keep the builtin's own error, no frame.
3. Combinators' argument order: `map_ok(f, r)` (function first,
   like reduce's `reduce(:op, x)`) -- proposed for consistency
   with the existing higher-order builtins.
