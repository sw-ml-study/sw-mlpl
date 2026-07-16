//! `depth(x)` introspection builtin (APL2 staging plan, Stage 1).
//!
//! Depth is the nesting level of a value. For MLPL's current flat
//! `DenseArray` (no boxed/nested arrays yet), depth follows APL2's
//! rule for simple arrays: a simple scalar has depth 0, and any
//! simple non-scalar array (vector, matrix, higher-rank) has depth 1.
//! Once nested arrays land (Stage 6) the same builtin will report
//! higher depths without changing these cases.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval_display(src: &str, env: &mut Environment) -> String {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap().to_string()
}

#[test]
fn depth_of_scalar_is_zero() {
    let mut env = Environment::new();
    assert_eq!(eval_display("depth(5)", &mut env), "0");
}

#[test]
fn depth_of_vector_is_one() {
    let mut env = Environment::new();
    assert_eq!(eval_display("depth([1, 2, 3])", &mut env), "1");
}

#[test]
fn depth_of_matrix_is_one() {
    let mut env = Environment::new();
    assert_eq!(
        eval_display("depth(reshape(iota(6), [2, 3]))", &mut env),
        "1"
    );
}

#[test]
fn depth_returns_a_scalar_rank_zero() {
    let mut env = Environment::new();
    // rank of the depth result is itself 0 -- depth yields a scalar.
    assert_eq!(eval_display("rank(depth([1, 2, 3]))", &mut env), "0");
}
