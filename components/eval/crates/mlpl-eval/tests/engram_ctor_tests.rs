//! `engram(...)` constructor (saga E2 step 1): spec validation at
//! the language surface, the five near-identity parameters, and
//! model introspection.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> Result<mlpl_eval::Value, mlpl_eval::EvalError> {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program_value(&stmts, env)
}

#[test]
fn constructor_builds_a_model_with_five_params() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(32, [2, 3], 4, 4096, 8, 7)").unwrap();
    let m = env.get_model("e").expect("model bound").clone();
    let params = m.params();
    assert_eq!(params.len(), 5, "memory + Wv + bv + Wg + bg");
    // memory table: [2 orders x 4 heads x 4096 slots, 8 head_dim]
    let mem = env.get(&params[0]).expect("memory param");
    assert_eq!(mem.shape().dims(), &[2 * 4 * 4096, 8]);
    assert!(mem.data().iter().all(|&v| v == 0.0), "memory starts zero");
    // value projection: [2*4*8 retrieved, 32 hidden]
    assert_eq!(env.get(&params[1]).unwrap().shape().dims(), &[64, 32]);
    // gate bias starts at -2 (nearly closed).
    let bg = env.get(&params[4]).unwrap();
    assert!(bg.data().iter().all(|&v| (v + 2.0).abs() < 1e-12));
}

#[test]
fn constructor_params_are_trainable_and_tagged() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(16, [2], 2, 256, 4, 1)").unwrap();
    let m = env.get_model("e").unwrap().clone();
    for p in m.params() {
        assert!(env.is_param(&p), "{p} must be a trainable param");
        assert!(env.get_tag(&p).is_some(), "{p} must be tagged");
    }
}

#[test]
fn invalid_specs_error_at_the_surface() {
    let mut env = Environment::new();
    let e = eval(&mut env, "engram(16, [1], 2, 256, 4, 1)").unwrap_err();
    assert!(format!("{e}").contains(">= 2"), "{e}");
    assert!(eval(&mut env, "engram(16, [2], 0, 256, 4, 1)").is_err());
    assert!(
        eval(&mut env, "engram(16, [2], 2, 256, 4)").is_err(),
        "arity"
    );
}

#[test]
fn describe_renders_the_engram_line() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(32, [2, 3], 4, 4096, 8, 7)").unwrap();
    let out = mlpl_eval::inspect(&mut env, ":describe e").expect("describe");
    assert!(
        out.contains("engram(hidden=32") && out.contains("slots=4096"),
        "describe should render the engram spec: {out}"
    );
}
