//! `write_stdout(bytes)` -- the non-seekable binary sink (the
//! ByteSink counterpart to read_stdin). Writes a rank-<=1 byte
//! array (0..=255) to process stdout and returns ok(count) / err.
//!
//! Like the print builtin, these tests pin the RETURN contract
//! (ok(count) on success, err Result / hard error on misuse) via
//! in-process eval; the actual stdout emission is verified by
//! running the binary against a script (the fs-append / connect
//! smoke covers the byte pipeline end to end).

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

#[test]
fn returns_ok_count_of_bytes_written() {
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "unwrap(write_stdout([72, 105]))"), 2.0);
    assert_eq!(
        scalar(&mut env, "unwrap(write_stdout([1, 2, 3, 4, 5]))"),
        5.0
    );
    // empty write is a valid no-op returning 0
    assert_eq!(scalar(&mut env, "unwrap(write_stdout([]))"), 0.0);
}

#[test]
fn invalid_input_is_an_err_result() {
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "is_ok(write_stdout([256]))"), 0.0);
    assert_eq!(scalar(&mut env, "is_ok(write_stdout([1, 3.5]))"), 0.0);
    assert_eq!(scalar(&mut env, "is_ok(write_stdout(\"hi\"))"), 0.0);
}

#[test]
fn wrong_arity_is_a_hard_error() {
    let mut env = Environment::new();
    assert!(eval_value(&mut env, "write_stdout()").is_err());
    assert!(eval_value(&mut env, "write_stdout([1], [2])").is_err());
}
