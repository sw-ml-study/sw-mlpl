//! `apply_engram(e, h, ids)` forward pass (saga E2 step 2): the
//! near-identity guarantee at init, memory actually flowing after a
//! table edit, and the shape/type error surface.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> Result<mlpl_array::DenseArray, mlpl_eval::EvalError> {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env)
}

#[test]
fn fresh_engram_is_exactly_identity() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(8, [2, 3], 2, 512, 4, 7)").unwrap();
    let out = eval(
        &mut env,
        "h = reshape(range(24), [3, 8]); apply_engram(e, h, [5, 6, 7]) - h",
    )
    .unwrap();
    assert!(
        out.data().iter().all(|&d| d == 0.0),
        "zero memory + zero b_v must be an EXACT no-op, got {:?}",
        out.data()
    );
}

#[test]
fn written_memory_rows_change_the_output() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(8, [2], 2, 64, 4, 7)").unwrap();
    // Poke the whole memory table to ones: every addressed row now
    // carries signal, so the output must move off the identity.
    let mem_name = env.get_model("e").unwrap().params()[0].clone();
    let mem = env.get(&mem_name).unwrap();
    let ones =
        mlpl_array::DenseArray::new(mem.shape().clone(), vec![1.0; mem.data().len()]).unwrap();
    env.set(mem_name, ones);
    let out = eval(
        &mut env,
        "h = reshape(range(16), [2, 8]); apply_engram(e, h, [1, 2]) - h",
    )
    .unwrap();
    assert!(
        out.data().iter().any(|&d| d != 0.0),
        "non-zero memory must influence the residual stream"
    );
}

#[test]
fn deterministic_across_calls() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(8, [2, 3], 2, 512, 4, 7)").unwrap();
    let a = eval(
        &mut env,
        "h = reshape(range(24), [3, 8]); apply_engram(e, h, [5, 6, 7])",
    )
    .unwrap();
    let b = eval(&mut env, "apply_engram(e, h, [5, 6, 7])").unwrap();
    assert_eq!(a.data(), b.data());
}

#[test]
fn shape_and_type_errors_are_loud() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(8, [2], 2, 64, 4, 7)").unwrap();
    // hidden width mismatch
    let e1 = eval(
        &mut env,
        "apply_engram(e, reshape(range(12), [2, 6]), [1, 2])",
    )
    .unwrap_err();
    assert!(format!("{e1}").contains("[T, 8]"), "{e1}");
    // ids length mismatch
    let e2 = eval(
        &mut env,
        "apply_engram(e, reshape(range(16), [2, 8]), [1, 2, 3])",
    )
    .unwrap_err();
    assert!(format!("{e2}").contains("same T"), "{e2}");
    // fractional ids
    let e3 = eval(
        &mut env,
        "apply_engram(e, reshape(range(16), [2, 8]), [1, 2.5])",
    )
    .unwrap_err();
    assert!(format!("{e3}").contains("integers"), "{e3}");
    // not an engram
    eval(&mut env, "m = linear(4, 4, 0)").unwrap();
    let e4 = eval(
        &mut env,
        "apply_engram(m, reshape(range(16), [2, 8]), [1, 2])",
    )
    .unwrap_err();
    assert!(format!("{e4}").contains("not an engram"), "{e4}");
}
