Add string literals to the parser and a Value type to the evaluator.

This is the foundation step for svg() and for future work with
LLM clients that need string arguments.

1. Parser: lex string literals
   - Double-quoted: "hello world"
   - Escape sequences: \" \\ \n
   - New TokenKind::StrLit(String)
   - New Expr::StrLit(String, Span)
   - Tests for lex + parse of various string literals

2. Evaluator: introduce Value enum in mlpl-eval
   - Value::Array(DenseArray)
   - Value::Str(String)
   - Change eval_program return type to Result<Value, EvalError>
   - Most existing call sites just wrap/unwrap Value::Array
   - Built-ins still take Vec<DenseArray> and return DenseArray
     for now (no built-in needs strings yet -- svg() comes next step)
   - Update all existing tests to wrap expected results in Value::Array

3. Display: impl Display for Value
   - Array: existing display
   - Str: just the contents (no quotes)

4. REPL (CLI + web): adapt to new Value return type
   - Print Value::Array as before
   - Print Value::Str as raw content

5. Quality gates pass

Allowed: crates/mlpl-parser, crates/mlpl-eval, apps/mlpl-repl,
crates/mlpl-wasm, apps/mlpl-web