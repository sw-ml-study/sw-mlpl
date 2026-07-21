//! `size(x)` and `tally(x)` introspection builtins (APL2 staging
//! plan, Stage 1).
//!
//! `size(x)` is the total number of elements (numel): the product of
//! the shape. A scalar has size 1. `tally(x)` is the length of the
//! leading axis -- the number of major cells (APL's monadic tally).
//! A scalar tallies to 1; a rank >= 1 array tallies to `shape[0]`.
//! Both return a rank-0 scalar so they compose with arithmetic.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval_display(src: &str, env: &mut Environment) -> String {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap().to_string()
}

#[test]
fn size_of_scalar_is_one() {
    let mut env = Environment::new();
    assert_eq!(eval_display("size(5)", &mut env), "1");
}

#[test]
fn size_of_vector_is_length() {
    let mut env = Environment::new();
    assert_eq!(eval_display("size([1, 2, 3, 4])", &mut env), "4");
}

#[test]
fn size_of_matrix_is_product_of_dims() {
    let mut env = Environment::new();
    assert_eq!(
        eval_display("size(reshape(iota(6), [2, 3]))", &mut env),
        "6"
    );
}

#[test]
fn size_returns_a_scalar_rank_zero() {
    let mut env = Environment::new();
    assert_eq!(eval_display("rank(size([1, 2, 3]))", &mut env), "0");
}

#[test]
fn tally_of_scalar_is_one() {
    let mut env = Environment::new();
    assert_eq!(eval_display("tally(5)", &mut env), "1");
}

#[test]
fn tally_of_vector_is_length() {
    let mut env = Environment::new();
    assert_eq!(eval_display("tally([1, 2, 3, 4])", &mut env), "4");
}

#[test]
fn tally_of_matrix_is_leading_axis() {
    let mut env = Environment::new();
    // reshape to [2, 3] -- two major cells (rows), so tally is 2
    // while size is 6.
    assert_eq!(
        eval_display("tally(reshape(iota(6), [2, 3]))", &mut env),
        "2"
    );
}

#[test]
fn tally_returns_a_scalar_rank_zero() {
    let mut env = Environment::new();
    assert_eq!(eval_display("rank(tally([1, 2, 3]))", &mut env), "0");
}
