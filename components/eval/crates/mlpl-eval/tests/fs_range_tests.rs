//! Bounded seek reads (demo-ml-utils large-file analysis):
//! read_bytes(path, offset, length) reads a slice without
//! materializing the whole file, and file_size(path) reports the
//! byte count from metadata. Sandboxed and Result-returning like
//! the rest of the fs API.

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
    let dir = std::env::temp_dir().join(format!("mlpl-range-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn env_with(dir: &std::path::Path) -> Environment {
    let mut env = Environment::new();
    env.fs_root = Some(dir.to_path_buf());
    env
}

fn seed(env: &mut Environment) {
    // 5 bytes: [10, 20, 30, 40, 50]
    eval_value(env, "write_bytes(\"d.bin\", [10, 20, 30, 40, 50])").unwrap();
}

#[test]
fn whole_file_read_is_unchanged() {
    let dir = sandbox("whole");
    let mut env = env_with(&dir);
    seed(&mut env);
    assert_eq!(
        scalar(
            &mut env,
            "equal([10, 20, 30, 40, 50], unwrap(read_bytes(\"d.bin\")))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn range_reads_a_slice() {
    let dir = sandbox("slice");
    let mut env = env_with(&dir);
    seed(&mut env);
    assert_eq!(
        scalar(
            &mut env,
            "equal([20, 30], unwrap(read_bytes(\"d.bin\", 1, 2)))"
        ),
        1.0
    );
}

#[test]
fn length_clamps_at_eof() {
    let dir = sandbox("clamp");
    let mut env = env_with(&dir);
    seed(&mut env);
    // from offset 3, ask for 100 -> only [40, 50] available
    assert_eq!(
        scalar(
            &mut env,
            "equal([40, 50], unwrap(read_bytes(\"d.bin\", 3, 100)))"
        ),
        1.0
    );
    // offset at or past EOF -> empty
    assert_eq!(
        scalar(&mut env, "tally(unwrap(read_bytes(\"d.bin\", 5, 10)))"),
        0.0
    );
    assert_eq!(
        scalar(&mut env, "tally(unwrap(read_bytes(\"d.bin\", 99, 10)))"),
        0.0
    );
}

#[test]
fn file_size_reports_byte_count() {
    let dir = sandbox("size");
    let mut env = env_with(&dir);
    seed(&mut env);
    assert_eq!(scalar(&mut env, "unwrap(file_size(\"d.bin\"))"), 5.0);
}

#[test]
fn bad_offset_or_length_type_is_a_hard_error() {
    let dir = sandbox("badtype");
    let mut env = env_with(&dir);
    seed(&mut env);
    assert!(eval_value(&mut env, "read_bytes(\"d.bin\", \"x\", 2)").is_err());
    assert!(eval_value(&mut env, "read_bytes(\"d.bin\", 1, 1.5)").is_err());
    assert!(eval_value(&mut env, "read_bytes(\"d.bin\", 0 - 1, 2)").is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_file_and_sandbox_escape_are_err_results() {
    let dir = sandbox("err");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "is_ok(read_bytes(\"nope.bin\", 0, 4))"),
        0.0
    );
    assert_eq!(
        scalar(&mut env, "is_ok(read_bytes(\"../evil.bin\", 0, 4))"),
        0.0
    );
    assert_eq!(scalar(&mut env, "is_ok(file_size(\"nope.bin\"))"), 0.0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_sandbox_surface_is_an_err_result() {
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "is_ok(read_bytes(\"d.bin\", 0, 4))"), 0.0);
    assert_eq!(scalar(&mut env, "is_ok(file_size(\"d.bin\"))"), 0.0);
}
