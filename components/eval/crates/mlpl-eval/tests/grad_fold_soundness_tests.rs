//! Soundness of grad's constant-fold fallback (demo-decision-model Q5). When a
//! call cannot be traced onto the tape, grad may fold it to a constant -- but
//! ONLY if it provably does not read a parameter's value. A `u:` function whose
//! BODY reads a global param must never be folded: that silently drops its
//! gradient term. The real tracing error must surface instead, naming the
//! unsupported form.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Vec<f64>, EvalError> {
    let prog = parse(&lex(src).expect("lex")).expect("parse");
    eval_program_value(&prog, env).map(|v| match v {
        Value::Array(a) => a.data().to_vec(),
        _ => Vec::new(),
    })
}

fn setup(src: &str) -> Environment {
    let mut env = Environment::new();
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        run(&mut env, line).expect("setup line");
    }
    env
}

const W3: &str = "W = param[3]\nW = [1.0, 2.0, 3.0]\nb = [1.0, 1.0, 1.0]\n\
                  def u:f(b) { reduce_add(abs(W) * b) }";

fn err_text(r: Result<Vec<f64>, EvalError>) -> String {
    match r {
        Ok(v) => panic!("expected an error, got {v:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn u_call_reading_param_in_body_is_not_folded_in_a_sum() {
    let mut env = setup(W3);
    // Previously returned [1, 1, 1]: the u:f term was folded to a constant
    // and only reduce_add(W) contributed -- a silently wrong gradient.
    let msg = err_text(run(&mut env, "grad(u:f(b) + reduce_add(W), W)"));
    assert!(
        msg.contains("abs"),
        "the real tracing error surfaces: {msg}"
    );
}

#[test]
fn u_call_reading_param_in_body_does_not_blame_the_param() {
    let mut env = setup(W3);
    let msg = err_text(run(&mut env, "grad(u:f(b), W)"));
    assert!(!msg.contains("does not depend"), "not blamed on W: {msg}");
    assert!(msg.contains("abs"), "names the unsupported function: {msg}");
}

#[test]
fn param_free_u_call_with_untraceable_body_still_folds() {
    let mut env = setup("W = param[3]\nW = [1.0, 2.0, 3.0]\ndef u:k() { if 1 { 2 } else { 3 } }");
    let g = run(&mut env, "grad(reduce_add(W) * u:k(), W)").expect("fold ok");
    assert_eq!(g, vec![2.0, 2.0, 2.0]);
}

#[test]
fn record_field_access_is_a_constant_leaf() {
    // Records are data: the field read is a constant, E gets the gradient.
    let mut env = setup("E = param[3]\nE = [1.0, 2.0, 3.0]\nr = {w: [1.0, 1.0, 1.0]}");
    let g = run(&mut env, "grad(reduce_add(E * r.w), E)").expect("field read");
    assert_eq!(g, vec![1.0, 1.0, 1.0]);
}

#[test]
fn record_argument_to_u_call_binds_for_field_reads() {
    let mut env = setup(
        "E = param[3]\nE = [1.0, 2.0, 3.0]\nr = {w: [1.0, 1.0, 1.0]}\n\
         def u:g(r) { reduce_add(E * r.w) }",
    );
    let g = run(&mut env, "grad(u:g(r), E)").expect("record argument");
    assert_eq!(g, vec![1.0, 1.0, 1.0]);
}

#[test]
fn an_unsupported_form_is_still_named() {
    let mut env = setup("E = param[3]\nE = [1.0, 2.0, 3.0]");
    let msg = err_text(run(&mut env, "grad(if 1 { reduce_add(E) } else { 0 }, E)"));
    assert!(msg.contains("an `if` expression"), "{msg}");
}

#[test]
fn optimizer_errors_when_a_requested_param_gets_no_gradient() {
    let mut env = setup("W = param[3]\nW = [1.0, 2.0, 3.0]\nV = param[3]\nV = [1.0, 1.0, 1.0]");
    let msg = err_text(run(
        &mut env,
        "adam(reduce_add(W * W), [W, V], 0.1, 0.9, 0.999, 0.00000001)",
    ));
    assert!(msg.contains("'V'") && msg.contains("no gradient"), "{msg}");
    let msg = err_text(run(
        &mut env,
        "momentum_sgd(reduce_add(W * W), [W, V], 0.1, 0.9)",
    ));
    assert!(msg.contains("'V'") && msg.contains("no gradient"), "{msg}");
}
