//! `equal` / `repr` -- mlplunit contract item 2: total structural
//! comparison that NEVER hard-errors on kind or shape mismatch,
//! and bounded deterministic diagnostics rendering.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn eq_scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap() {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn arrays_records_results_and_mismatches() {
    let mut env = Environment::new();
    assert_eq!(eq_scalar(&mut env, "equal([1, 2, 3], [1, 2, 3])"), 1.0);
    assert_eq!(eq_scalar(&mut env, "equal([1, 2, 3], [1, 2, 4])"), 0.0);
    assert_eq!(
        eq_scalar(&mut env, "equal([1, 2], reshape([1, 2], [2, 1]))"),
        0.0,
        "shape distinguishes"
    );
    assert_eq!(
        eq_scalar(
            &mut env,
            "equal({a: ok([1, 2]), b: 3}, {a: ok([1, 2]), b: 3})"
        ),
        1.0
    );
    assert_eq!(
        eq_scalar(&mut env, "equal({a: ok(1)}, {a: err(1)})"),
        0.0,
        "ok vs err"
    );
    assert_eq!(
        eq_scalar(&mut env, "equal(1, ok(1))"),
        0.0,
        "kind mismatch is FALSE, not an error"
    );
    assert_eq!(eq_scalar(&mut env, "equal(\"hi\", \"hi\")"), 1.0);
    assert_eq!(eq_scalar(&mut env, "equal(\"hi\", \"ho\")"), 0.0);
}

#[test]
fn labels_distinguish_arrays() {
    let mut env = Environment::new();
    eval_value(&mut env, "a : [row, col] = reshape(range(4), [2, 2])").unwrap();
    eval_value(&mut env, "b = reshape(range(4), [2, 2])").unwrap();
    assert_eq!(eq_scalar(&mut env, "equal(a, b)"), 0.0);
    assert_eq!(eq_scalar(&mut env, "equal(a, a)"), 1.0);
}

#[test]
fn repr_is_bounded_and_shape_bearing() {
    let mut env = Environment::new();
    let Value::Str(r) = eval_value(&mut env, "repr(reshape(range(6), [2, 3]))").unwrap() else {
        panic!("repr must return a string")
    };
    assert!(r.contains("array[2, 3]"), "{r}");
    let Value::Str(big) = eval_value(&mut env, "repr(range(10000))").unwrap() else {
        panic!()
    };
    assert!(big.len() < 500, "bounded, got {} chars", big.len());
    assert!(big.contains("10000 values"), "{big}");
    let Value::Str(rec) = eval_value(&mut env, "repr({b: ok(2), a: \"x\"})").unwrap() else {
        panic!()
    };
    assert_eq!(rec, "{a: \"x\", b: ok(array[] [2])}");
}
