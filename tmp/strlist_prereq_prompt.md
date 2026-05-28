Saga 29 step 002 (inserted): value-strlist-and-list-literal.

Why: docs/milestone-vit.md step 001 specifies
`load_preloaded("pets_tiny")` to return
`{X: [200, 3, 64, 64], Y: [200], names: [str]}` — a record whose
`names` field is a list of file basenames. Step 001
(`value-record-and-field-access`) added `Value::Record` so the
record half of that spec is achievable, but MLPL has no
list-of-strings value type today (verified:
crates/mlpl-eval/src/value.rs variants are Array, Str, Model,
Tokenizer, BuiltinRef, DeviceTensor, Record). The `[...]` array
literal currently builds a `DenseArray` of `f64`, full stop;
strings inside `[...]` error at eval time. This step adds
`Value::StrList` so the next step
(`load-images-and-pets-tiny`) can populate the `names` field
without resorting to a synthetic index-keyed record.

Scope (one PR / one step):

1. Value (crates/mlpl-eval/src/value.rs):
   - new `Value::StrList { items: Vec<String> }` variant.
   - Display impl prints `["a", "b", "c"]` (double-quoted,
     comma-separated). Empty list prints `[]`.
   - `value_kind()` returns `"string-list"`.

2. Eval (crates/mlpl-eval/src/eval.rs):
   - dispatch `Expr::ArrayLit` first: if every element evaluates
     to `Value::Str`, return `Value::StrList`. If every element
     evaluates to a scalar `Value::Array`, fall through to the
     existing `eval_array_lit` path. Mixed element kinds error
     cleanly with `EvalError::MixedArrayLitElements { kinds:
     Vec<&'static str> }` — a new variant.
   - The early-return must run BEFORE the existing `match expr`
     ArrayLit dispatch (which forces every element to a
     DenseArray and currently panics or surface-errors on
     `Value::Str` elements).

3. Cross-crate match exhaustiveness (every `match value` site
   that pre-existed and gained a Record arm in step 001):
   - crates/mlpl-eval/src/* — value.rs Display + value_kind,
     env.rs (if it has Value-discriminating helpers), eval.rs
     value-shaped early-returns.
   - crates/mlpl-trace/src/* — JSON event serialization. StrList
     serializes as `{"kind": "string-list", "items": [...]}` or
     similar; pick one and document.
   - crates/mlpl-serve/src/handlers.rs — eval handler response
     shape + inspect handler. Whatever Record returns, StrList
     mirrors.
   - services/mlpl-mlx-serve/src/handlers.rs — clean reject (the
     MLX peer wire does not encode string lists).
   - crates/mlpl-lower-rs — Record arm in the unsupported tail;
     same treatment for StrList.

4. Tests (crates/mlpl-eval/tests/strlist_tests.rs and
   crates/mlpl-parser/tests/strlist_parse_tests.rs):
   - `["a", "b", "c"]` evaluates to `Value::StrList` with 3 items.
   - `[]` continues to evaluate to an empty `DenseArray`
     (existing behavior preserved; an empty `[]` is ambiguous
     and we keep the array path for back-compat).
   - `[1, 2, 3]` continues to evaluate to a `DenseArray` of
     shape `[3]` (no regression).
   - `["a", 1]` errors with `MixedArrayLitElements` and the
     error names both kinds.
   - `["a"]` evaluates to a `Value::StrList` with 1 item.
   - Display round-trip: `["a", "b"]` → `["a", "b"]`.
   - Cross-crate: `trace.json` records a StrList event without
     panicking; `mlpl-serve` `value_kind` returns
     `"string-list"`.
   - Record-of-strlist: `{ names: ["a", "b"] }` evaluates to a
     `Value::Record` whose `names` field is a `Value::StrList`.
   - FieldAccess on a record-of-strlist returns the StrList:
     `let r = { names: ["a", "b"] }; r.names` is a StrList of 2.

5. Accessors (minimum viable; pick ONE for this step, defer
   the others to follow-ups):
   - `len(xs)` where `xs` is a `Value::StrList` returns its
     length as a scalar `DenseArray`. The existing `len` builtin
     already exists for arrays — extend it to StrList. (If `len`
     does not yet exist, ship a new `list_len(xs)` builtin
     instead; do not block on adding a general `len`.)
   - Indexing into a StrList (`names[3]` or `index(names, 3)`)
     is OUT OF SCOPE for this step; defer to a follow-up step or
     to the consumer step. The pets_tiny demo can iterate names
     via Display today; programmatic indexing arrives later.

6. Contract: contracts/eval-contract README and
   contracts/parser-contract README updated in the same commit
   with one paragraph each on StrList and the `[...]`
   disambiguation rule.

7. Docs: add a "String lists" section to docs/glossary.md
   (right after the "Records" section step 001 added); one
   sentence in docs/saga.md noting StrList landed as Saga 29
   prerequisite step 002.

TDD: Red the tests first. Quality gates: cargo test workspace,
cargo clippy --all-targets --all-features -- -D warnings,
cargo fmt --all -- --check, sw-checklist held or lowered,
markdown-checker on touched docs. /mw-cp checkpoint.

Disk hygiene: eval + parser + trace + serve crate work. Scoped
builds first (`cargo test -p mlpl-eval -p mlpl-parser -p
mlpl-trace -p mlpl-serve`); workspace test once at the end.
Check `df -h /` before starting; if `target/` is over 10 GB,
`cargo clean` first. No new dependencies expected.

Out of scope (followups, not this step):
- StrList indexing (`names[i]`, `index(names, i)`) — wait for
  the consumer step or a dedicated indexing step.
- StrList destructuring / pattern matching.
- `concat_str_list(a, b)`, `push_str(xs, s)`, or any mutation.
- Heterogeneous lists (`[1, "a", true]`); explicit step 003
  decision is that mixed element types are an error.
- StrList in autograd / tape / MLX peer — those layers reject
  cleanly via the cross-crate match arms above.

Why this is one step not three: every cross-crate match needs
the StrList arm to keep the workspace building; splitting the
arms across multiple commits leaves the tree in a broken state.
Same lesson as 21.5 step 011's BuiltinRef and step 001's
Record cross-workspace drift.
