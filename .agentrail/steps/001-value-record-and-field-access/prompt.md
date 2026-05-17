Saga 29 step 001 (inserted): value-record-and-field-access.

Why: docs/milestone-vit.md Step 001 says
load_preloaded("pets_tiny") returns {X: [200, 3, 64, 64], Y:
[200], names: [str]} -- a record. But Value has no Record /
Tuple variant today (verified: crates/mlpl-eval/src/value.rs
variants are Array, Str, Model, Tokenizer, BuiltinRef,
DeviceTensor only). And the parser has no Dot token, no
FieldAccess / RecordLit AST node, no precedent for
disambiguating `{X: e1}` (record literal) from `{ stmt; }`
(block used by repeat/train/for/experiment/device). This step
adds records as a first-class Value so load_preloaded("pets_tiny")
can return one in the next step.

Scope (one PR / one step):

1. Lexer (crates/mlpl-parser/src/lex.rs): add TokenKind::Dot
   for field-access syntax. Single-char `.` distinct from float
   literals (which already lex as one token).

2. AST (crates/mlpl-parser/src/ast.rs):
   - new `Expr::RecordLit { fields: Vec<(String, Expr)>, span }`
   - new `Expr::FieldAccess { receiver: Box<Expr>, field: String,
     span }`

3. Parser: parse `{ ident : expr (, ident : expr)* }` as
   RecordLit. Disambiguate from `{ stmt }` blocks by one-token
   lookahead: `{` followed by ident-then-colon-then-non-newline-
   non-equals is a record; everything else is a block. The
   colon is already used by annotation syntax (`x : [batch,
   feat] = ...`) on the LHS of assign; record literal is on the
   RHS so context disambiguates. Parse `expr.ident` postfix as
   FieldAccess; lower precedence than function call so
   `f(x).y` works.

4. Value (crates/mlpl-eval/src/value.rs):
   - new `Value::Record { fields: BTreeMap<String, Value> }`
     (BTreeMap for deterministic key ordering in display +
     trace + JSON serialization)
   - Display impl prints `{ k1: <v1>, k2: <v2> }` with each
     value rendered via the existing per-variant Display.

5. Eval (crates/mlpl-eval/src/eval.rs):
   - dispatch Expr::RecordLit by evaluating each field expr
     and collecting into a BTreeMap
   - dispatch Expr::FieldAccess: if receiver is Value::Record,
     look up the key; unknown key -> EvalError::FieldNotFound
     { record_keys, requested }
   - non-record receiver -> EvalError::FieldOnNonRecord {
     receiver_kind, field }

6. Cross-crate match exhaustiveness (every `match value` site
   that pre-existed):
   - crates/mlpl-eval/src/* (display, dispatch, trace_emit, etc.)
   - crates/mlpl-trace/src/* (JSON event serialization)
   - crates/mlpl-serve/src/* (inspect handler, eval handler
     response shape)
   - services/mlpl-mlx-serve/src/handlers.rs (the same match
     that gained BuiltinRef in 21.5 step 011 -- add Record arm
     that rejects with a clean error since the MLX peer wire
     does not encode records)

7. Tests (crates/mlpl-eval/tests/record_tests.rs):
   - record literal evaluates to Value::Record
   - field access returns the right value
   - unknown field name returns a tutoring error
   - nested record + nested field access
   - record-of-arrays round-trips through eval / display
   - parser rejects malformed record (e.g., `{X: 1, Y: }`)
   - block-vs-record disambiguation: `repeat 3 { x = 1 }` is a
     block; `{ X: 1 }` at expr position is a record

8. Contract: contracts/parser-contract.md and
   contracts/eval-contract.md updated in the same commit.

9. Docs: add a "Records" section to docs/glossary.md; one
   sentence in docs/saga.md Saga 29 narrative noting records
   landed as a prerequisite.

TDD: Red the tests first. Quality gates: cargo test workspace,
cargo clippy -D warnings, cargo fmt --all -- --check, sw-checklist
held or lowered, markdown-checker on touched docs. /mw-cp
checkpoint.

Disk hygiene: this is parser + eval crate work, scoped builds
should work (cargo test -p mlpl-parser -p mlpl-eval); only run
the workspace test once at the end. Check df -h / before
starting; if target/ is over 10 GB, cargo clean first.

Out of scope (followups, not this step):
- Record destructuring in let-bindings (`let {X, Y} = r`).
- Record update / spread syntax (`{ ..r, X: new_x }`).
- Pattern matching on records.
- Records-as-trace-events (Saga 4 trace JSON already encodes
  Value variants; adding Record is mechanical extension).
- Records as builtin argument types beyond load_preloaded;
  callers can keep using positional args for now.

Why this is one step not three: every cross-crate match needs
the Record arm to keep the workspace building; splitting the
arms across multiple commits leaves the tree in a broken state.
Same lesson as 21.5 step 011's BuiltinRef cross-workspace drift.
