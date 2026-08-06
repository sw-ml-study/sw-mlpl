//! each(f, v) and table(f, a, b) -- the first APL2/BQN
//! higher-order builtins over function references: elementwise
//! application with shape preserved, and the outer product
//! over a function.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn arr(env: &mut Environment, src: &str) -> mlpl_array::DenseArray {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a,
        other => panic!("expected array from {src}, got {other:?}"),
    }
}

#[test]
fn each_applies_a_user_fn_per_element_shape_preserved() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:sq(x) { x * x }").unwrap();
    let a = arr(&mut env, "each(:u:sq, [1, 2, 3, 4])");
    assert_eq!(a.data(), &[1.0, 4.0, 9.0, 16.0]);
    // Rank 2: same shape out.
    let a = arr(&mut env, "each(:u:sq, reshape(range(6), [2, 3]))");
    assert_eq!(a.shape().dims(), &[2, 3]);
    assert_eq!(a.data(), &[0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
    // Scalar in, scalar out.
    let a = arr(&mut env, "each(:u:sq, 7)");
    assert_eq!(a.data(), &[49.0]);
}

#[test]
fn each_accepts_builtin_references_too() {
    let mut env = Environment::new();
    let a = arr(&mut env, "each(:sqrt, [1, 4, 9])");
    assert_eq!(a.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn table_is_the_outer_product_over_f() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:times(a, b) { a * b }").unwrap();
    // APL2:  A ∘.× B   /  BQN:  a ×⌜ b
    let a = arr(&mut env, "table(:u:times, [1, 2, 3], [10, 20])");
    assert_eq!(a.shape().dims(), &[3, 2]);
    assert_eq!(a.data(), &[10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
}

#[test]
fn empty_inputs_stay_empty() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:sq(x) { x * x }").unwrap();
    let a = arr(&mut env, "each(:u:sq, fill([0], 0))");
    assert_eq!(a.shape().dims(), &[0]);
    eval_value(&mut env, "def u:add2(a, b) { a + b }").unwrap();
    let a = arr(&mut env, "table(:u:add2, fill([0], 0), [1, 2])");
    assert_eq!(a.shape().dims(), &[0, 2]);
}

#[test]
fn non_scalar_returns_error_loudly_with_the_index() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:vec(x) { [x, x] }").unwrap();
    let e = eval_value(&mut env, "each(:u:vec, [1, 2])").unwrap_err();
    assert!(
        e.contains("each") && e.contains("scalar") && e.contains("0"),
        "{e}"
    );
}

#[test]
fn misuse_is_structured() {
    let mut env = Environment::new();
    let e = eval_value(&mut env, "each(1, [1, 2])").unwrap_err();
    assert!(e.contains("each") && e.contains("reference"), "{e}");
    let e = eval_value(&mut env, "table(:add, 1, [1, 2])").unwrap_err();
    assert!(e.contains("table") && e.contains("rank-1"), "{e}");
}
