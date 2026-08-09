//! append_bytes(path, bytes) -- the bounded incremental SINK: the
//! output half of the ByteSource/ByteSink contract. Appends a
//! rank-<=1 byte array (0..=255) to a file (creating it if
//! absent), returning ok(count) / err. Position = file_size,
//! flush is implicit per append. Sandboxed, Result-based like the
//! other byte fs ops.

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
    let dir = std::env::temp_dir().join(format!("mlpl-append-{}-{tag}", std::process::id()));
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
fn append_creates_then_reads_back() {
    let dir = sandbox("new");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "unwrap(append_bytes(\"a.bin\", [1, 2, 3]))"),
        3.0
    );
    assert_eq!(
        scalar(&mut env, "equal([1, 2, 3], unwrap(read_bytes(\"a.bin\")))"),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn appends_grow_the_file_incrementally() {
    let dir = sandbox("grow");
    let mut env = env_with(&dir);
    eval_value(&mut env, "append_bytes(\"g.bin\", [1, 2])").unwrap();
    assert_eq!(scalar(&mut env, "unwrap(file_size(\"g.bin\"))"), 2.0);
    eval_value(&mut env, "append_bytes(\"g.bin\", [3, 4])").unwrap();
    assert_eq!(scalar(&mut env, "unwrap(file_size(\"g.bin\"))"), 4.0);
    assert_eq!(
        scalar(
            &mut env,
            "equal([1, 2, 3, 4], unwrap(read_bytes(\"g.bin\")))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bounded_read_transform_append_composes() {
    let dir = sandbox("xform");
    let mut env = env_with(&dir);
    // source of 5 bytes; copy it to a dest one 2-byte chunk at a
    // time via bounded reads + appends -- never the whole file.
    eval_value(&mut env, "write_bytes(\"src.bin\", [10, 20, 30, 40, 50])").unwrap();
    eval_value(
        &mut env,
        "append_bytes(\"dst.bin\", unwrap(read_bytes(\"src.bin\", 0, 2)))",
    )
    .unwrap();
    eval_value(
        &mut env,
        "append_bytes(\"dst.bin\", unwrap(read_bytes(\"src.bin\", 2, 2)))",
    )
    .unwrap();
    eval_value(
        &mut env,
        "append_bytes(\"dst.bin\", unwrap(read_bytes(\"src.bin\", 4, 2)))",
    )
    .unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal([10, 20, 30, 40, 50], unwrap(read_bytes(\"dst.bin\")))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn invalid_input_is_an_err_result() {
    let dir = sandbox("bad");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "is_ok(append_bytes(\"a.bin\", [256]))"),
        0.0
    );
    assert_eq!(
        scalar(&mut env, "is_ok(append_bytes(\"a.bin\", \"hi\"))"),
        0.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sandbox_escape_and_no_surface_are_err_results() {
    let dir = sandbox("esc");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "is_ok(append_bytes(\"../evil.bin\", [1]))"),
        0.0
    );
    let mut bare = Environment::new();
    assert_eq!(
        scalar(&mut bare, "is_ok(append_bytes(\"x.bin\", [1]))"),
        0.0
    );
    std::fs::remove_dir_all(&dir).ok();
}
