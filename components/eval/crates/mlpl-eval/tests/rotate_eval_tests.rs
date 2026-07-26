//! `rotate(x, k, axis)` through the full lex/parse/eval pipeline
//! (Game of Life saga step 1): plain dispatch, negative k via
//! `0 - 1` (MLPL has no unary minus), and argument errors.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval_ok(src: &str) -> mlpl_eval::Value {
    let toks = lex(src).expect("lex");
    let prog = parse(&toks).expect("parse");
    let mut env = Environment::new();
    eval_program_value(&prog, &mut env).expect("eval")
}

fn eval_err(src: &str) -> String {
    let toks = lex(src).expect("lex");
    let prog = parse(&toks).expect("parse");
    let mut env = Environment::new();
    format!(
        "{:?}",
        eval_program_value(&prog, &mut env).expect_err("expected eval error")
    )
}

fn as_data(v: mlpl_eval::Value) -> Vec<f64> {
    match v {
        mlpl_eval::Value::Array(a) => a.data().to_vec(),
        other => panic!("expected array, got {other:?}"),
    }
}

#[test]
fn rotate_vector_left() {
    assert_eq!(
        as_data(eval_ok("rotate([1, 2, 3, 4], 1, 0)")),
        vec![2.0, 3.0, 4.0, 1.0]
    );
}

#[test]
fn rotate_vector_right_via_zero_minus_one() {
    assert_eq!(
        as_data(eval_ok("v = [1, 2, 3, 4]; rotate(v, 0 - 1, 0)")),
        vec![4.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn rotate_matrix_columns() {
    assert_eq!(
        as_data(eval_ok("rotate(reshape(iota(6), [2, 3]), 1, 1)")),
        vec![1.0, 2.0, 0.0, 4.0, 5.0, 3.0]
    );
}

#[test]
fn rotate_axis_out_of_range_is_an_error() {
    let msg = eval_err("rotate([1, 2, 3], 1, 4)");
    assert!(!msg.is_empty());
}

#[test]
fn rotate_wrong_arity_is_an_error() {
    let msg = eval_err("rotate([1, 2, 3], 1)");
    assert!(msg.contains("Arity") || msg.contains("arity") || msg.contains("expected"));
}

#[test]
fn life_step_via_rotate_neighbors() {
    // A blinker oscillates: vertical bar -> horizontal bar. The
    // 8-neighbor sum built from rotate() is the Life engine the
    // demo will ship; assert one full step here.
    let src = r#"
G = reshape([0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,0,0,0], [5, 5])
u = rotate(G, 1, 0)
d = rotate(G, 0 - 1, 0)
l = rotate(G, 1, 1)
r = rotate(G, 0 - 1, 1)
ul = rotate(u, 1, 1)
ur = rotate(u, 0 - 1, 1)
dl = rotate(d, 1, 1)
dr = rotate(d, 0 - 1, 1)
N = u + d + l + r + ul + ur + dl + dr
gt(eq(N, 3) + G * eq(N, 2), 0)
"#;
    let got = as_data(eval_ok(src));
    let expected = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(got, expected);
}
