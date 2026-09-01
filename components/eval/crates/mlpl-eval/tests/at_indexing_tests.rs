//! `at(v, i)` -- the 2-argument vector-indexing convenience for
//! `take(v, 0, i)` (../emufpga ask 3), so reading one element of a
//! vector no longer needs the explicit axis. Returns a scalar for a
//! rank-1 input.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Vec<f64> {
    let toks = lex(src).unwrap();
    let stmts = parse(&toks).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap().data().to_vec()
}

#[test]
fn at_reads_a_vector_element_as_a_scalar() {
    assert_eq!(eval("at([10, 20, 30], 1)"), vec![20.0]);
    assert_eq!(
        eval("order = [3, 1, 4, 1]\nat(order, 0) + at(order, 2)"),
        vec![7.0]
    );
}

#[test]
fn at_is_take_on_axis_zero() {
    // at(v, i) == take(v, 0, i) for a vector.
    assert_eq!(eval("at(iota(5), 3)"), eval("take(iota(5), 0, 3)"));
}

#[test]
fn at_result_broadcasts_against_a_vector() {
    // The scalar from `at` meets a vector directly (length-1 broadcast).
    assert_eq!(eval("at([2, 5, 9], 0) * [1, 2, 3]"), vec![2.0, 4.0, 6.0]);
}
