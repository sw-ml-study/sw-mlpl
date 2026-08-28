//! Qualified (`namespace:name`) first-class references. Generalizes
//! the old `:u:name`-only rule so library / provider namespaces
//! (`:result:zip`, `:math:double`) participate in `call`, `each`,
//! partial application, and callbacks -- the reference parity the
//! demo-mlpl-libraries package API needs. Bare `:name` refs stay
//! builtin operator references.

use mlpl_eval::{Environment, Value};

fn eval(src: &str) -> Value {
    let toks = mlpl_parser::lex(src).expect("lex");
    let stmts = mlpl_parser::parse(&toks).expect("parse");
    let mut env = Environment::new();
    mlpl_eval::eval_program_value(&stmts, &mut env).expect("eval")
}

fn scalars(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("expected an array, got {other:?}"),
    }
}

#[test]
fn call_through_a_library_namespace_reference() {
    let v = eval("def math:double(x) { \"Double x.\" x * 2 }\ncall(:math:double, 21)\n");
    assert_eq!(scalars(&v), vec![42.0]);
}

#[test]
fn each_over_a_provider_namespace_reference() {
    let v = eval("def result:sq(x) { \"Square x.\" x * x }\neach(:result:sq, [2, 3, 4])\n");
    assert_eq!(scalars(&v), vec![4.0, 9.0, 16.0]);
}

#[test]
fn u_namespace_still_resolves_as_before() {
    let v = eval("def u:inc(x) { \"Increment.\" x + 1 }\ncall(:u:inc, 10)\n");
    assert_eq!(scalars(&v), vec![11.0]);
}

#[test]
fn bare_builtin_reference_still_reduces() {
    // `:add` has no inner colon -> stays a builtin operator reference.
    let v = eval("reduce(:add, iota(5))\n");
    assert_eq!(scalars(&v), vec![10.0]);
}
