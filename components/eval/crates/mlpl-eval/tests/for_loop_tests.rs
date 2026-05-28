//! Tests for the `for <ident> in <expr> { body }` construct
//! (Saga 12 step 003).
//!
//! Semantics mirror `train`: each iteration binds `ident` to a
//! row-slice of the source array, runs the body, and captures the
//! body's final value into a `last_rows` vector in the environment.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval_and_get(src: &str, probe: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap();
    // After the loop body, re-run a probe expression in the same env.
    let tokens = lex(probe).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, &mut env).unwrap()
}

#[test]
fn for_sums_each_row() {
    // reshape(iota(6), [3, 2]) -> rows [[0,1],[2,3],[4,5]]
    // reduce_add of each row -> [1, 5, 9]
    let src = "for row in reshape(iota(6), [3, 2]) { reduce_add(row) }";
    let last = eval_and_get(src, "last_rows");
    assert_eq!(last.shape().dims(), &[3]);
    assert_eq!(last.data(), &[1.0, 5.0, 9.0]);
}

#[test]
fn for_binds_rank_minus_one_slice() {
    // shape of each row slice of a [3, 2] array is [2].
    let src = "for row in reshape(iota(6), [3, 2]) { shape(row) }";
    let last = eval_and_get(src, "last_rows");
    // last_rows is a [3, 1] matrix (each iteration returns a rank-1
    // shape vector of length 1, stacked).
    assert_eq!(last.shape().dims(), &[3, 1]);
    assert_eq!(last.data(), &[2.0, 2.0, 2.0]);
}

#[test]
fn for_over_rank1_binds_scalar() {
    // iota(4) -> rows are scalars 0, 1, 2, 3.
    // row + 10 -> 10, 11, 12, 13.
    let src = "for v in iota(4) { v + 10 }";
    let last = eval_and_get(src, "last_rows");
    assert_eq!(last.shape().dims(), &[4]);
    assert_eq!(last.data(), &[10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn for_preserves_non_axis0_labels_on_slices() {
    // x : [batch, feat] = [3, 2]; inside the loop, `row` is a
    // rank-1 slice that should carry the axis-1 label ("feat").
    // After the loop, `row` still holds the last iteration's slice.
    let tokens = lex("x : [batch, feat] = reshape(iota(6), [3, 2])\n\
         for row in x { v = row }\n\
         labels(row)")
    .unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let v = mlpl_eval::eval_program_value(&stmts, &mut env).unwrap();
    assert_eq!(v, mlpl_eval::Value::Str("feat".into()));
}

#[test]
fn for_source_rank0_errors() {
    let src = "for v in 42 { v }";
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let r = eval_program(&stmts, &mut env);
    assert!(r.is_err(), "rank-0 source must error");
}

#[test]
fn for_empty_source_yields_empty_last_rows() {
    let src = "e = reshape(iota(0), [0])\nfor v in e { v + 1 }";
    let last = eval_and_get(src, "last_rows");
    // Zero iterations -> empty last_rows vector.
    assert_eq!(last.shape().dims(), &[0]);
}

// -- Parser surface --

#[test]
fn parser_produces_for_variant() {
    use mlpl_parser::Expr;
    let tokens = lex("for row in iota(3) { row }").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1);
    match &stmts[0] {
        Expr::For {
            binding,
            source: _,
            body,
            ..
        } => {
            assert_eq!(binding, "row");
            assert_eq!(body.len(), 1);
        }
        other => panic!("expected Expr::For, got {other:?}"),
    }
}

#[test]
fn repeat_and_train_still_parse() {
    // Regression: the new `for` keyword must not break existing
    // loop constructs.
    parse(&lex("repeat 3 { x = 1 }").unwrap()).unwrap();
    parse(&lex("train 5 { x = 1 }").unwrap()).unwrap();
}

// -- Compile path --

#[test]
fn for_is_unsupported_in_lower_rs() {
    use mlpl_lower_rs::{LowerError, lower};
    let stmts = parse(&lex("for row in iota(3) { row }").unwrap()).unwrap();
    let r = lower(&stmts);
    assert!(matches!(r, Err(LowerError::Unsupported(_))), "got {r:?}");
}
