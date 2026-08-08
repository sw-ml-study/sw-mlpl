//! Sandboxed raw-byte fs builtins (demo-algorithms serialization,
//! raw bytes / I/O): read_bytes / write_bytes -- Result-returning,
//! contained by the environment's sandbox root, a byte an f64 in
//! 0..256 (the tokenize_bytes / bit-ops convention). Domain
//! violations (out-of-range or non-integer cells, a non-array
//! byte arg) are LOUD hard errors naming the culprit.

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

fn sandbox(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlpl-bytes-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn env_with(dir: &std::path::Path) -> Environment {
    let mut env = Environment::new();
    env.fs_root = Some(dir.to_path_buf());
    env
}

#[test]
fn byte_round_trip_is_exact() {
    let dir = sandbox("round");
    let mut env = env_with(&dir);
    eval_value(&mut env, "write_bytes(\"data.bin\", [104, 105, 0, 255])").unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal([104, 105, 0, 255], unwrap(read_bytes(\"data.bin\")))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_bytes_returns_ok_one() {
    let dir = sandbox("okone");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "unwrap(write_bytes(\"x.bin\", [1, 2, 3]))"),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn read_bytes_missing_file_is_err_result() {
    let dir = sandbox("missing");
    let mut env = env_with(&dir);
    // ok field of the Result is 0 (err), not a hard error.
    assert_eq!(scalar(&mut env, "is_ok(read_bytes(\"nope.bin\"))"), 0.0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn out_of_range_byte_is_a_hard_error() {
    let dir = sandbox("range");
    let mut env = env_with(&dir);
    let e = eval_value(&mut env, "write_bytes(\"a.bin\", [1, 256])").unwrap_err();
    assert!(e.contains("256"), "error should name the culprit: {e}");
    assert!(eval_value(&mut env, "write_bytes(\"b.bin\", [1, -1])").is_err());
    assert!(eval_value(&mut env, "write_bytes(\"c.bin\", [1, 3.5])").is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn non_array_byte_arg_is_a_hard_error() {
    let dir = sandbox("nonarr");
    let mut env = env_with(&dir);
    assert!(eval_value(&mut env, "write_bytes(\"s.bin\", \"hi\")").is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sandbox_escape_is_an_err_result() {
    let dir = sandbox("escape");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "is_ok(write_bytes(\"../evil.bin\", [1]))"),
        0.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_sandbox_surface_is_an_err_result() {
    let mut env = Environment::new(); // no fs_root (browser-like)
    assert_eq!(scalar(&mut env, "is_ok(read_bytes(\"x.bin\"))"), 0.0);
    assert_eq!(scalar(&mut env, "is_ok(write_bytes(\"x.bin\", [1]))"), 0.0);
}

#[test]
fn binary_json_round_trip_composes() {
    let dir = sandbox("json");
    let mut env = env_with(&dir);
    // encode a record to JSON, store as raw bytes, read back, decode, re-parse.
    eval_value(
        &mut env,
        "write_bytes(\"r.json\", tokenize_bytes(to_json({a: 1, b: 2})))",
    )
    .unwrap();
    // the bytes on disk decode back to the exact JSON text...
    assert_eq!(
        scalar(
            &mut env,
            "equal(to_json({a: 1, b: 2}), decode_bytes(unwrap(read_bytes(\"r.json\"))))"
        ),
        1.0
    );
    // ...and re-parse to a record (parse_json returns a Result).
    assert_eq!(
        scalar(
            &mut env,
            "equal(\"record\", type_of(unwrap(parse_json(decode_bytes(unwrap(read_bytes(\"r.json\")))))))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}
