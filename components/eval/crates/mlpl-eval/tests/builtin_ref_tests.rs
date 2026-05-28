//! Higher-order `reduce(:op, x[, axis])` via `:foo` BuiltinRef
//! syntax. Covers lexer + parser + eval + dispatch end-to-end.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn try_run(src: &str, env: &mut Environment) -> Result<DenseArray, EvalError> {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env)
}

#[test]
fn reduce_add_named_form() {
    let mut env = Environment::new();
    let r = run("v = [1, 2, 3, 4]\nreduce(:add, v)\n", &mut env);
    assert_eq!(r.data()[0], 10.0);
}

#[test]
fn reduce_plus_operator_form() {
    let mut env = Environment::new();
    let r = run("v = [1, 2, 3, 4]\nreduce(:+, v)\n", &mut env);
    assert_eq!(r.data()[0], 10.0);
}

#[test]
fn reduce_max() {
    let mut env = Environment::new();
    let r = run("v = [3, 1, 4, 1, 5, 9, 2, 6]\nreduce(:max, v)\n", &mut env);
    assert_eq!(r.data()[0], 9.0);
}

#[test]
fn reduce_min() {
    let mut env = Environment::new();
    let r = run("v = [3, 1, 4, 1, 5, 9, 2, 6]\nreduce(:min, v)\n", &mut env);
    assert_eq!(r.data()[0], 1.0);
}

#[test]
fn reduce_mul_star_alias() {
    let mut env = Environment::new();
    let r = run("v = [2, 3, 4]\nreduce(:*, v)\n", &mut env);
    assert_eq!(r.data()[0], 24.0);
}

#[test]
fn reduce_with_axis() {
    let mut env = Environment::new();
    env.set(
        "M".into(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let r = run("reduce(:add, M, 1)\n", &mut env);
    assert_eq!(r.shape().dims(), &[2]);
    assert_eq!(r.data(), &[6.0, 15.0]);
}

#[test]
fn builtin_ref_can_be_assigned_and_reused() {
    let mut env = Environment::new();
    let r = run(
        "f = :max\nv = [3, 1, 4, 1, 5, 9, 2, 6]\nreduce(f, v)\n",
        &mut env,
    );
    assert_eq!(r.data()[0], 9.0);
}

#[test]
fn user_variable_does_not_shadow_builtin_ref() {
    // `add = 42` binds a scalar to `add` in env.vars. The
    // BuiltinRef namespace is separate, so `:add` still works.
    let mut env = Environment::new();
    let r = run("add = 42\nv = [1, 2, 3]\nreduce(:add, v)\n", &mut env);
    assert_eq!(r.data()[0], 6.0);
}

#[test]
fn passing_string_to_reduce_errors_cleanly() {
    let mut env = Environment::new();
    let err = try_run("v = [1, 2, 3]\nreduce(\"add\", v)\n", &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("builtin reference") && msg.contains(":add"),
        "expected guidance toward :add form: {msg}"
    );
}

#[test]
fn reduce_unknown_op_errors_with_curated_set() {
    let mut env = Environment::new();
    let err = try_run("v = [1, 2, 3]\nreduce(:xor, v)\n", &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown op") && msg.contains(":xor"),
        "expected unknown-op error: {msg}"
    );
}

#[test]
fn passing_scalar_to_reduce_errors_cleanly() {
    // `add = 42; reduce(add, v)` -- `add` resolves to the
    // scalar 42, not a BuiltinRef. The reduce dispatch must
    // reject with a clear guidance error.
    let mut env = Environment::new();
    let err = try_run("add = 42\nv = [1, 2, 3]\nreduce(add, v)\n", &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("builtin reference"),
        "expected guidance toward :add form: {msg}"
    );
}
