//! `reduce(:op, x, axis)` axis selection (demo-ml-utils C3 + C4):
//! multi-axis positional lists and named-axis (label) reduction, so a
//! convolution can contract its whole receptive field in one call.

use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn a_scalar_axis_still_reduces_one_axis() {
    // Backward compatibility with the pre-C3 single-axis form.
    let y = eval("reduce(:add, reshape(range(6), [2, 3]), 1)").unwrap();
    assert_eq!(y.data(), &[3.0, 12.0]);
}

#[test]
fn a_vector_of_positions_reduces_several_axes() {
    // [2,3,4] range, reduce axes 1 and 2 -> [2], each the sum of its 12.
    let y = eval("reduce(:add, reshape(range(24), [2, 3, 4]), [1, 2])").unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![2]));
    assert_eq!(y.data(), &[66.0, 210.0]);
}

#[test]
fn a_string_names_a_labeled_axis() {
    let m = "label(reshape(range(6), [2, 3]), [\"a\", \"b\"])";
    assert_eq!(
        eval(&format!("reduce(:add, {m}, \"a\")")).unwrap().data(),
        &[3.0, 5.0, 7.0]
    );
    assert_eq!(
        eval(&format!("reduce(:add, {m}, \"b\")")).unwrap().data(),
        &[3.0, 12.0]
    );
}

#[test]
fn a_comma_list_of_names_reduces_several_labeled_axes() {
    let y =
        eval("reduce(:add, label(reshape(range(6), [2, 3]), [\"a\", \"b\"]), \"a,b\")").unwrap();
    assert_eq!(y.data(), &[15.0]);
}

#[test]
fn the_surviving_axes_keep_their_labels() {
    // Reducing the channel axis of [c,y,x] leaves [y,x] labeled, so named
    // reduces chain (e.g. contract channel then kernel axes by name).
    let src = "labels(reduce(:add, label(reshape(range(24), [2, 3, 4]), \
               [\"c\", \"y\", \"x\"]), \"c\"))";
    let stmts = parse(&lex(src).unwrap()).unwrap();
    let v = mlpl_eval::eval_program_value(&stmts, &mut Environment::new()).unwrap();
    assert_eq!(v, mlpl_eval::Value::Str("y,x".into()));
}

#[test]
fn an_unknown_label_is_an_error() {
    assert!(
        eval("reduce(:add, label(reshape(range(6), [2, 3]), [\"a\", \"b\"]), \"zzz\")").is_err()
    );
}

#[test]
fn a_named_axis_on_an_unlabeled_array_is_an_error() {
    assert!(eval("reduce(:add, reshape(range(6), [2, 3]), \"a\")").is_err());
}
