//! moe-microscope finding F1: softmax arity must match between eager
//! evaluation and the grad/adam tape. The tape's softmax is unary (last
//! axis); eager required an explicit axis, so the same loss could not be
//! written once. Eager softmax(x) now defaults to the last axis too.

use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn softmax_one_arg_defaults_to_last_axis() {
    // softmax(x) with no axis: a probability distribution over the last axis.
    let y = eval("softmax([1.0, 2.0, 3.0])").unwrap();
    let sum: f64 = y.data().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "softmax sums to 1: {:?}",
        y.data()
    );
    assert!(y.data()[2] > y.data()[0], "monotone: {:?}", y.data());
}

#[test]
fn softmax_one_arg_equals_explicit_last_axis() {
    // softmax(x) == softmax(x, last) for a rank-2 input (per-row).
    let implicit = eval("softmax(reshape(range(6), [2, 3]))").unwrap();
    let explicit = eval("softmax(reshape(range(6), [2, 3]), 1)").unwrap();
    assert_eq!(implicit.data(), explicit.data());
}
