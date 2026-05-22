Scripting saga step 005: add `while` + `break` + `continue`. Refactor-first, then add.

**This step is BOTH a language feature AND a ratchet-down refactor.** The order matters: refactor `crates/mlpl-eval/src/eval.rs` BEFORE adding the new control-flow surface, per the stricter rule in `docs/code_metrics.md` ("Do not add new logic to an over-limit function or module. First extract responsibilities into named pure helpers, then add").

`crates/mlpl-eval/src/eval.rs` is currently ~995 lines (the FAIL floor is 500; the code_metrics gate is ~25 LOC per function in a 5-fn module, i.e. ~125 LOC per file). It is the single biggest sw-checklist FAIL in the repo and the natural home for the new while/break/continue eval rules. Refactoring it first kills two birds: ratchet-down ~3-7 FAILs in one move, and create the right shape for the new code to land in.

## Phase A: refactor eval.rs (do this FIRST)

Read `docs/code_metrics.md` end-to-end before starting -- especially sections 1-4 (design principles, refactoring triggers, module shape, file naming).

Per the file-naming convention, eval.rs's current responsibilities break into:

- **`eval.rs`** (after refactor): facade only. `pub fn eval_expr(...)`, `pub fn eval_program(...)`, `pub fn eval_program_value(...)`, `pub fn eval_program_traced(...)`. Maybe 50-100 lines.
- **`eval_dispatch.rs`** (or `dispatch.rs`): the big match in `eval_expr` over `Expr` variants. Extract to a dispatch helper that calls per-category handlers below.
- **`eval_lookup.rs`** (or `lookup.rs`): Ident lookup, Assign handling.
- **`eval_ctrlflow.rs`** (or `ctrlflow.rs`): all loop / branch eval rules. Currently `Expr::Repeat` / `Expr::Train` / `Expr::For` / `Expr::Experiment` / `Expr::Device` / new `Expr::If` (from step 004) / new `Expr::While` / new break+continue handling. THIS is where the new step-005 logic lands.
- **`eval_literals.rs`** (or `literals.rs`): IntLit / FloatLit / StrLit / BuiltinRef / ArrayLit / RecordLit / FieldAccess / Result construction.
- **`eval_intercepts.rs`** (or `intercepts.rs`): the if-let chain of FnCall intercepts at the head of `eval_expr` (print/eprint, to_number/to_int/env, args/list_get, ok/err, is_ok/unwrap/etc., svg, grad, apply, etc.). Currently this is ~half the file.

mlpl-eval is ALREADY at 37 modules (max 7) -- adding more files makes the module-count FAIL count WORSE per-module-count, but the file-LOC + module-fn-count fails on eval.rs will retire, and the per-file fn-count and per-file LOC of the split files will all be small wins. Net should be positive ratchet-down (rough estimate: -3 to -7 FAILs).

Process: extract ONE module at a time. After each extraction, run `cargo test -p mlpl-eval --release` to confirm behavior unchanged. Do NOT rewrite logic during the split -- preserve byte-for-byte where possible.

Target after refactor:

- eval.rs at <=200 lines (well under 500 file-LOC fail)
- eval.rs with <=5 production fns (well under 7 fn-count fail) -- the public eval_program* entry points only
- Each new sibling file <=125 lines / <=5 fns
- All existing tests still green

## Phase B: add while / break / continue (after refactor lands)

Surface forms:
  while cond { body }      -- loop until cond is non-truthy
  break                    -- exit innermost loop
  break val                -- exit innermost loop with `val` as the loop value
  continue                 -- skip to next iteration

`break` and `continue` work inside ANY loop form: while, repeat, train, for. `break` outside a loop is a parse-time or eval-time error (parser-side check preferred for early diagnostics).

`while` loop value semantics: the value of the last body expression of the last iteration (matches `repeat` semantics). `break val` overrides this and makes the entire while loop evaluate to `val`. A while-with-no-iterations evaluates to a scalar 0.

Truthiness rules for `cond` are the same as for `if` (see step 004): non-zero scalar or `Ok(_)` Result.

Implementation:

1. New AST nodes in `crates/mlpl-parser/src/ast.rs`: `Expr::While { cond, body, span }`, `Expr::Break { value: Option<Box<Expr>>, span }`, `Expr::Continue { span }`. Add to the `span()` accessor and the `Display` impl (the `fmt_compound` extraction from step 004 makes this easy).
2. Lexer keywords: `while`, `break`, `continue`. Extend the keyword match in `lex_ident` per step 004's pattern.
3. Parser: while is a statement-form like repeat (handled by `parse_statement`); break and continue are atoms (handled by `parse_atom`). Add corresponding helpers in parser.rs or a new `stmts_loops.rs` if the parser module is at its function-count cap.
4. Eval: the new while/break/continue rules go in the freshly-extracted `eval_ctrlflow.rs` (NOT in eval.rs). Break and continue need a control-flow SIGNAL that propagates UP through the eval chain. Two designs to pick between:
   - Extend `EvalError` with `BreakSignal { value: Option<Value> }` and `ContinueSignal` variants. Loops catch them and convert to their own value / iteration boundary; non-loop scopes propagate them up. This piggybacks on the existing ? operator.
   - Use a separate `EvalControl` enum returned alongside `Value`. Cleaner type story but requires touching every eval helper signature.

   Recommend: option 1 (EvalError signal variants). Document in `crates/mlpl-eval/src/error.rs` that break/continue are NOT really errors but use the error channel for propagation. Add a helper `bare_loop_value(EvalResult<Value>) -> EvalResult<Value>` that loops call to catch their own break/continue. Test that break out of a function call boundary is a clear error message (since MLPL has no user-defined functions yet, this surface is small).

## TDD (do this regardless of phase order)

Parser tests:
- `while 1 { 42 }` parses to Expr::While
- `while cond { break }` parses
- `while cond { continue }` parses
- `break val` parses (with value)
- `break` outside any loop must be a parse-time error OR a parse-time check that the parser tracks loop-depth. (If parser doesn't track depth, eval-time error is acceptable but document the choice.)

Eval tests:
- while loop terminates when cond becomes falsy
- break exits the loop and returns the last body value
- `break val` returns val
- continue skips to next iteration
- break inside nested while only exits the innermost loop
- break inside repeat / train / for works the same way
- break outside any loop is an error
- A while loop with `Ok(_)`-conditional terminates when the condition becomes Err.

## Quality gates

cargo test workspace; cargo clippy --workspace --all-targets --all-features -- -D warnings; cargo fmt; markdown-checker on any docs touched; sw-checklist MUST RATCHET DOWN by at least 5 fails -- the Phase A refactor is the primary vehicle. If after both phases the count hasn't dropped by 5, IDENTIFY WHY and either bundle one more retirement or document the exception per CLAUDE.md.

## Docs

- `docs/lang-reference.md`: extend the Scripting section with `while` / `break` / `continue` entries.
- `docs/glossary.md`: add `while loop`, `break`, `continue` entries.
- Closes audit finding #23 -- update `docs/language-status.md` accordingly.

After this step ships, MLPL has the full classical control-flow surface (if + while + break + continue) and scripts can write adaptive loops (train until validation loss stops dropping, generate tokens until EOS, etc.).
