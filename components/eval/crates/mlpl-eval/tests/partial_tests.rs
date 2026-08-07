//! Partial application (docs/combinators-design.md): under-
//! application of call() returns a Partial VALUE, exact arity
//! executes, excess arguments apply left-associatively -- the
//! bridge from higher-order functions to combinatory logic.

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

const AVIARY: &str = "
def u:I(x) { x }
def u:K(x, y) { x }
def u:B(x, y, z) { call(x, call(y, z)) }
def u:S(x, y, z) { call(call(x, z), call(y, z)) }
def u:M(x) { call(x, x) }
def u:sq(x) { x * x }
def u:inc(x) { x + 1 }
";

#[test]
fn under_application_returns_a_callable_partial() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    // K 5 is a THING; (K 5) 99 is 5.
    eval_value(&mut env, "k5 = call(:u:K, 5)").unwrap();
    assert_eq!(scalar(&mut env, "call(k5, 99)"), 5.0);
    // Reusable: the partial applies again to a different y.
    assert_eq!(scalar(&mut env, "call(k5, 1)"), 5.0);
}

#[test]
fn staged_and_saturated_bluebird_agree() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    let staged = scalar(&mut env, "call(call(call(:u:B, :u:sq), :u:inc), 4)");
    let saturated = scalar(&mut env, "call(:u:B, :u:sq, :u:inc, 4)");
    assert_eq!(staged, 25.0);
    assert_eq!(saturated, 25.0);
}

#[test]
fn excess_arguments_apply_left_associatively() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    // I has arity 1; call(I, K, 5, 99) = ((I K) 5) 99 = (K 5) 99 = 5.
    assert_eq!(scalar(&mut env, "call(:u:I, :u:K, 5, 99)"), 5.0);
}

#[test]
fn sk_basis_derives_identity() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    // S K K x == x -- the research's acceptance criterion.
    eval_value(&mut env, "skk = call(:u:S, :u:K, :u:K)").unwrap();
    assert_eq!(scalar(&mut env, "call(skk, 42)"), 42.0);
}

#[test]
fn mockingbird_self_application_works() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    // M I = I I = I: a partial-free self-application path.
    let v = eval_value(&mut env, "call(:u:M, :u:I)").unwrap();
    assert!(
        matches!(&v, Value::BuiltinRef { name } | Value::UserFnRef { name } if name == "u:I"),
        "M I should be I itself: {v:?}"
    );
}

#[test]
fn partials_flow_through_storage_args_and_returns() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    // Record field.
    eval_value(&mut env, "nest = {five: call(:u:K, 5)}").unwrap();
    assert_eq!(scalar(&mut env, "call(nest.five, 0)"), 5.0);
    // Function argument + return.
    eval_value(
        &mut env,
        "def u:apply_twice(f, x) { call(f, call(f, x)) }\n\
         def u:make_adder(n) { call(:u:addn, n) }\n\
         def u:addn(n, x) { n + x }",
    )
    .unwrap();
    eval_value(&mut env, "add3 = u:make_adder(3)").unwrap();
    assert_eq!(scalar(&mut env, "call(add3, 10)"), 13.0);
    assert_eq!(scalar(&mut env, "u:apply_twice(add3, 0)"), 6.0);
}

#[test]
fn partials_work_in_the_hof_quartet() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    eval_value(&mut env, "def u:addn(n, x) { n + x }").unwrap();
    eval_value(&mut env, "add10 = call(:u:addn, 10)").unwrap();
    let v = eval_value(&mut env, "each(add10, [1, 2, 3])").unwrap();
    assert!(
        matches!(&v, Value::Array(a) if a.data() == [11.0, 12.0, 13.0]),
        "{v:?}"
    );
    assert_eq!(scalar(&mut env, "atop(add10, :u:sq, 3)"), 19.0);
}

#[test]
fn repr_equal_and_errors_speak_partials() {
    let mut env = Environment::new();
    eval_value(&mut env, AVIARY).unwrap();
    let v = eval_value(&mut env, "repr(call(:u:B, :u:sq))").unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s.contains("partial") && s.contains("u:B") && s.contains("1 of 3")),
        "{v:?}"
    );
    assert_eq!(scalar(&mut env, "equal(call(:u:K, 5), call(:u:K, 5))"), 1.0);
    assert_eq!(scalar(&mut env, "equal(call(:u:K, 5), call(:u:K, 6))"), 0.0);
    // Builtin under-application is a tutoring error.
    let e = eval_value(&mut env, "call(:mean)").unwrap_err();
    assert!(e.contains("wrap") || e.contains("u:"), "{e}");
    // Excess application onto a non-callable is loud.
    let e = eval_value(&mut env, "call(:u:I, 5, 6)").unwrap_err();
    assert!(e.contains("apply") || e.contains("call"), "{e}");
}
