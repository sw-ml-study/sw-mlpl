//! axis-naming-unification: `reduce(:op, x, ["a", "b"])` -- the bracketed
//! list of axis NAMES, the form `label` has always accepted but `reduce`
//! rejected. It now resolves through the shared `mlpl_axes::AxisSpec`, so
//! names, a comma-string, and integer indices are interchangeable.

use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn bracketed_name_list_reduces_all_named_axes() {
    // The reported inconsistency: this used to error "expected an array
    // value, got a string". reshape(range(6),[2,3]) summed over both named
    // axes -> scalar 15.
    let y = eval("reduce(:add, label(reshape(range(6), [2, 3]), [\"a\", \"b\"]), [\"a\", \"b\"])")
        .unwrap();
    assert_eq!(y.data(), &[15.0]);
}

#[test]
fn bracketed_single_name_reduces_one_axis() {
    // reduce axis "a" (=0) of [[0,1,2],[3,4,5]] -> column sums [3,5,7].
    let y =
        eval("reduce(:add, label(reshape(range(6), [2, 3]), [\"a\", \"b\"]), [\"a\"])").unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![3]));
    assert_eq!(y.data(), &[3.0, 5.0, 7.0]);
}

#[test]
fn bracketed_names_match_the_comma_string_and_index_forms() {
    // ["b","c"] == "b,c" == positions [1,2] on a [2,3,4] range -> [66,210].
    let names = eval(
        "reduce(:add, label(reshape(range(24), [2, 3, 4]), [\"a\", \"b\", \"c\"]), [\"b\", \"c\"])",
    )
    .unwrap();
    let comma =
        eval("reduce(:add, label(reshape(range(24), [2, 3, 4]), [\"a\", \"b\", \"c\"]), \"b,c\")")
            .unwrap();
    let idx = eval("reduce(:add, reshape(range(24), [2, 3, 4]), [1, 2])").unwrap();
    assert_eq!(names.data(), &[66.0, 210.0]);
    assert_eq!(names.data(), comma.data());
    assert_eq!(names.data(), idx.data());
}

#[test]
fn unknown_name_in_a_list_errors() {
    assert!(
        eval("reduce(:add, label(reshape(range(6), [2, 3]), [\"a\", \"b\"]), [\"zzz\"])").is_err()
    );
}

#[test]
fn named_list_on_unlabeled_array_errors() {
    assert!(eval("reduce(:add, reshape(range(6), [2, 3]), [\"a\"])").is_err());
}
