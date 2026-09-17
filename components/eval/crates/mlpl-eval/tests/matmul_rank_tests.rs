//! RS4 (../reasoning-from-scratch): rank-3 (batched) matmul is rejected with
//! a clear, actionable message instead of a misleading index/panic error.
//! matmul is 2-D only ([m,k] @ [k,n] or [k]); batched matmul is a later ask.

use mlpl_array::DenseArray;
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

fn err_msg(src: &str) -> String {
    format!("{}", eval(src).unwrap_err())
}

#[test]
fn rank3_matmul_errors_clearly_eager() {
    let msg = err_msg("matmul(reshape(range(24), [2, 3, 4]), reshape(range(24), [2, 4, 3]))");
    assert!(msg.contains("matmul"), "names the op: {msg}");
    assert!(
        msg.to_lowercase().contains("rank") && !msg.contains("index has"),
        "actionable rank message, not the index error: {msg}"
    );
}

#[test]
fn rank3_matmul_errors_clearly_in_grad() {
    let mut env = Environment::new();
    env.set_param(
        "w".into(),
        DenseArray::new(
            mlpl_array::Shape::new(vec![2, 3, 4]),
            (0..24).map(|n| n as f64).collect(),
        )
        .unwrap(),
    );
    let toks = lex("grad(sum(matmul(w, reshape(range(12), [4, 3]))), w)").unwrap();
    let stmts = parse(&toks).unwrap();
    let msg = format!("{}", eval_program(&stmts, &mut env).unwrap_err());
    assert!(msg.contains("matmul"), "grad path names the op: {msg}");
    assert!(
        !msg.contains("index has"),
        "not the misleading index error: {msg}"
    );
}

#[test]
fn valid_2d_matmul_unaffected() {
    let out = eval("matmul(reshape(range(6), [2, 3]), reshape(range(6), [3, 2]))").unwrap();
    assert_eq!(out.shape().dims(), &[2, 2]);
}

#[test]
fn matrix_vector_still_works() {
    let out = eval("matmul(reshape(range(6), [2, 3]), range(3))").unwrap();
    assert_eq!(out.shape().dims(), &[2]);
}
