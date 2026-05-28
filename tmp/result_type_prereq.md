Saga 29 step 012 (inserted prereq): value-result-type.

Why: the user asked for a Result<val, error> value type so
error-tolerant ops (upload(), file I/O, parse, etc.) can
return success / failure as a first-class value rather than
bubbling out of the REPL as raw error messages. Two concrete
follow-ups need it:

- Step 013 (`:upload x` REPL command) -- user picks a file
  or cancels. Cancellation should bind `x = Err("cancelled")`
  rather than leaving `x` undefined.
- Step 014 (held-out test set demo) -- nice-to-have but
  optional.

Once the Result type exists, the upload + future error-
tolerant builtins can adopt it. The current button-based
upload UX (step 011 follow-up) continues to work unchanged
because it side-effects bind a name, not return a value.

Scope (one PR / one step):

1. Value variant (crates/mlpl-eval/src/value.rs):
   - new `Value::Result { ok: bool, payload: Box<Value> }`.
     The `ok` flag discriminates success vs error; `payload`
     can be any Value (typically Array for success, Str for
     error messages).
   - Display: `Ok(<inner>)` or `Err(<inner>)`.
   - `value_kind` returns `"result"`.

2. Cross-crate match arms (same drift trap as Record /
   StrList expansions):
   - mlpl-eval/src/eval.rs Assign arm (store the Result
     value in env).
   - mlpl-serve/src/handlers.rs value_kind.
   - services/mlpl-mlx-serve/src/handlers.rs reject arm.
   - mlpl-lower-rs unsupported tail.

3. Env storage (crates/mlpl-eval/src/env.rs):
   - new `results: HashMap<String, (bool, Value)>` namespace,
     mirroring the `records` and `string_lists` namespaces
     from steps 001 / 002.
   - `set_result(name, ok, payload)` and
     `get_result(name) -> Option<&(bool, Value)>` helpers.

4. Constructors (eval-side, since payload is a Value):
   - `ok(value)` builtin: wrap any Value as `Ok(_)`.
   - `err(value)` builtin: wrap any Value as `Err(_)`.

5. Accessors:
   - `is_ok(r)` / `is_err(r)`: scalar 0/1 DenseArray.
   - `unwrap(r)`: returns the inner value if Ok; clean
     `UnwrapOnErr` error otherwise.
   - `unwrap_or(r, default)`: returns inner if Ok, else
     `default`. Type-discriminates -- the default is
     accepted as-is.
   - `err_message(r)`: returns the inner Value (typically
     Str) if Err; clean error if Ok.

6. New EvalError variant: `UnwrapOnErr { message: String }`
   so unwrap() failures surface with the inner err payload.

7. Tests (crates/mlpl-eval/tests/result_tests.rs):
   - constructor + Display round trips.
   - is_ok / is_err / unwrap / unwrap_or / err_message
     forward and error paths.
   - record-of-result (a Record field can hold a Result).
   - Result-of-record (a Result payload can be a Record).
   - Result-of-strlist (Result payload can be StrList).
   - rebind a Result variable in env.
   - cross-crate: serialize through trace.json (no panic),
     serve handler `value_kind` returns "result".

8. Contracts (contracts/eval-contract README): document
   Result semantics + the unwrap / unwrap_or / err_message
   accessors + the UnwrapOnErr error variant.

9. docs/glossary.md: new "Result (value type)" entry.
   docs/lang-reference.md: rows for ok / err / is_ok /
   is_err / unwrap / unwrap_or / err_message.
   README count drift bumped.

Quality gates: cargo test (workspace), clippy -D warnings,
fmt, markdown-checker, sw-checklist held or lowered.
/mw-cp checkpoint. Commit + push before agentrail complete.

Out of scope (followups):
- Pattern matching: `match r { Ok x => ..., Err e => ... }`
  syntax. Useful but requires parser + AST changes;
  separate step. For now `is_ok` / `unwrap` is the surface.
- `Result<...>` type annotations in the assign-with-type-tag
  syntax. Same scope expansion.
- `try { ... }` block that wraps body in a Result. Useful
  for the upload builtin but layered on top of the basic
  Result variant; separate step.
- Automatic conversion from EvalError to Err -- today every
  error still bubbles to the REPL. A future `try { }` block
  catches EvalError and produces an Err result.

Why this is one step: every cross-crate match arm needs to
gain the Result variant for the workspace to build. Adding
the constructors + accessors in the same commit keeps the
type usable from day one.
