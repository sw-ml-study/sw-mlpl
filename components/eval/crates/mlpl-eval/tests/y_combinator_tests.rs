//! The fixed-point combinator over partials (the Y-combinator
//! examples in the Combinators demo): recursion without a step
//! function naming itself. No new language feature -- partials
//! supply the delay a strict-language fixed point needs.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

#[test]
fn self_application_recurses_without_named_recursion() {
    // fact_step never names itself -- it calls the `self` it was
    // handed (the mockingbird move: call(self, self, ...)).
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:fact_step(self, n) { if gt(n, 1) { n * call(self, self, n - 1) } else { 1 } }",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "call(:u:fact_step, :u:fact_step, 5)"), 120.0);
}

const FIX: &str = "
def u:fix(f, v) { call(f, call(:u:fix, f), v) }
def u:fact_body(rec, n) { if gt(n, 1) { n * call(rec, n - 1) } else { 1 } }
def u:fib_body(rec, n) { if lt(n, 2) { n } else { call(rec, n - 1) + call(rec, n - 2) } }
";

#[test]
fn fix_ties_the_knot_for_a_clean_step() {
    // fact_body / fib_body never reference themselves; fix
    // supplies recursion. The partial call(:u:fix, f) is the
    // delayed self-reference that lets this run in a STRICT
    // language (the classic lazy Y would diverge here).
    let mut env = Environment::new();
    eval_value(&mut env, FIX).unwrap();
    let fact = scalar(&mut env, "call(call(:u:fix, :u:fact_body), 6)");
    assert_eq!(fact, 720.0);
    let fib = scalar(&mut env, "call(call(:u:fix, :u:fib_body), 10)");
    assert_eq!(fib, 55.0);
    // The tied knot is an ordinary value: bind it, reuse it.
    eval_value(&mut env, "fact = call(:u:fix, :u:fact_body)").unwrap();
    assert_eq!(scalar(&mut env, "call(fact, 4)"), 24.0);
    assert_eq!(scalar(&mut env, "call(fact, 5)"), 120.0);
}

#[test]
fn a_base_case_that_returns_immediately_does_not_recurse() {
    let mut env = Environment::new();
    eval_value(&mut env, FIX).unwrap();
    assert_eq!(scalar(&mut env, "call(call(:u:fix, :u:fact_body), 1)"), 1.0);
    assert_eq!(scalar(&mut env, "call(call(:u:fix, :u:fib_body), 0)"), 0.0);
}
