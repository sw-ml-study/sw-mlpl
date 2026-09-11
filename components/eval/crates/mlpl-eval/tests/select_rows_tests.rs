//! storage-layout-viz: `select_rows(table, keys)` -- vectorized keyed row
//! lookup. Given a record mapping each key to a numeric row and a string
//! list of keys, gather the rows into an [N, C] matrix. This is the
//! primitive that turns a region-kind list into an [N,4] RGBA palette
//! selection (and any categorical -> attribute-row mapping).

use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn gathers_palette_rows_by_kind() {
    let src = "palette = {header: [0.5, 0.5, 0.5, 1.0], \
                catalog: [0.2, 0.6, 1.0, 1.0], \
                image: [1.0, 0.4, 0.2, 1.0]}; \
               select_rows(palette, [\"catalog\", \"image\", \"header\"])";
    let y = eval(src).unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![3, 4]));
    assert_eq!(
        y.data(),
        &[0.2, 0.6, 1.0, 1.0, 1.0, 0.4, 0.2, 1.0, 0.5, 0.5, 0.5, 1.0]
    );
}

#[test]
fn repeated_keys_repeat_rows() {
    let src = "p = {a: [1.0, 2.0], b: [3.0, 4.0]}; select_rows(p, [\"a\", \"a\", \"b\"])";
    let y = eval(src).unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![3, 2]));
    assert_eq!(y.data(), &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn missing_key_errors() {
    assert!(eval("select_rows({a: [1.0, 2.0]}, [\"b\"])").is_err());
}

#[test]
fn ragged_rows_error() {
    assert!(eval("select_rows({a: [1.0, 2.0], b: [3.0]}, [\"a\", \"b\"])").is_err());
}

#[test]
fn non_record_table_errors() {
    assert!(eval("select_rows([1.0, 2.0], [\"a\"])").is_err());
}
