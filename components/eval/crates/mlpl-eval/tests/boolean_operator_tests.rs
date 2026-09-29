//! `and` / `or` / `not` (microgpt-mlpl request #6): Python-style keywords.
//! Precedence, loosest first: `or` < `and` < `not` < comparisons <
//! arithmetic. On scalars they short-circuit (the right side is not
//! evaluated when the left decides); on arrays they are elementwise over
//! 0/1 masks (nonzero = true). Results are 0 / 1. Inside `grad` they are
//! stop-gradient masks, like the comparisons.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Value, EvalError> {
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        last = eval_program_value(&parse(&lex(line).unwrap()).unwrap(), env)?;
    }
    Ok(last)
}

fn nums(src: &str) -> Vec<f64> {
    match run(&mut Environment::new(), src) {
        Ok(Value::Array(a)) => a.data().to_vec(),
        other => panic!("{src}: {other:?}"),
    }
}

#[test]
fn precedence_matches_python() {
    assert_eq!(nums("1 or 0 and 0"), vec![1.0]); // 1 or (0 and 0)
    assert_eq!(nums("(1 or 0) and 0"), vec![0.0]);
    assert_eq!(nums("not 0 and 0"), vec![0.0]); // (not 0) and 0
    assert_eq!(nums("not 1 < 0"), vec![1.0]); // not (1 < 0)
    assert_eq!(nums("1 + 1 > 1 and 3 == 3"), vec![1.0]);
    assert_eq!(nums("not not 5"), vec![1.0]);
}

#[test]
fn scalars_short_circuit() {
    let boom = "def u:boom() { \"never called\"; undefined_fn(1) }\n";
    assert_eq!(nums(&format!("{boom}0 and u:boom()")), vec![0.0]);
    assert_eq!(nums(&format!("{boom}2 or u:boom()")), vec![1.0]);
    assert!(run(&mut Environment::new(), &format!("{boom}1 and u:boom()")).is_err());
    assert_eq!(nums("3 and 4"), vec![1.0]); // 0/1 result, not Python's operand
}

#[test]
fn arrays_are_elementwise_masks_with_broadcasting() {
    assert_eq!(nums("[1, 0, 2] and [1, 1, 0]"), vec![1.0, 0.0, 0.0]);
    assert_eq!(nums("[0, 0, 5] or 0"), vec![0.0, 0.0, 1.0]);
    assert_eq!(nums("not [0, 3]"), vec![1.0, 0.0]);
    assert_eq!(
        nums("[[1, 0], [0, 1]] and [1, 1]"),
        vec![1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        nums("x = [1, 5, 9]\ncompress(x > 2 and x < 8, x)"),
        vec![5.0]
    );
}

#[test]
fn control_flow_conditions() {
    let got = nums(
        "i = 0\ndone = 0\nwhile i < 10 and not done { i = i + 1; if i == 3 { done = 1 } else { 0 } }\ni",
    );
    assert_eq!(got, vec![3.0]);
}

#[test]
fn inside_grad_they_are_masks() {
    let src = "W = param[4]\nW = [1.0, 2.0, 3.0, 4.0]\nc = [0, 1, 2, 3]\n\
               grad(reduce_add(W * (c > 0 and c < 3 or c == 3)), W)";
    assert_eq!(nums(src), vec![0.0, 1.0, 1.0, 1.0]);
}
