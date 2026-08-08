//! Atomic sandboxed writes (demo-algorithms serialization,
//! atomic writes): write_atomic(path, value) writes a string or
//! byte array via a temp file + rename, so a reader sees the old
//! file or the whole new one -- never a torn write. Result-
//! returning and sandbox-contained like the other fs ops.

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
    let dir = std::env::temp_dir().join(format!("mlpl-atomic-{}-{tag}", std::process::id()));
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
fn string_round_trip_via_read_text() {
    let dir = sandbox("str");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "unwrap(write_atomic(\"a.txt\", \"hello\"))"),
        1.0
    );
    assert_eq!(
        scalar(&mut env, "equal(\"hello\", unwrap(read_text(\"a.txt\")))"),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn byte_array_round_trip_via_read_bytes() {
    let dir = sandbox("bytes");
    let mut env = env_with(&dir);
    eval_value(&mut env, "write_atomic(\"a.bin\", [1, 2, 254, 255])").unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal([1, 2, 254, 255], unwrap(read_bytes(\"a.bin\")))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn overwrites_existing_file_wholesale() {
    let dir = sandbox("over");
    let mut env = env_with(&dir);
    eval_value(&mut env, "write_atomic(\"a.txt\", \"first-longer\")").unwrap();
    eval_value(&mut env, "write_atomic(\"a.txt\", \"second\")").unwrap();
    assert_eq!(
        scalar(&mut env, "equal(\"second\", unwrap(read_text(\"a.txt\")))"),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn leaves_no_temp_file_behind() {
    let dir = sandbox("notemp");
    let mut env = env_with(&dir);
    eval_value(&mut env, "write_atomic(\"data.json\", \"{}\")").unwrap();
    // only the target file remains -- no stray temp artifact.
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec!["data.json".to_string()],
        "unexpected files: {names:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn out_of_range_byte_is_a_hard_error() {
    let dir = sandbox("range");
    let mut env = env_with(&dir);
    let e = eval_value(&mut env, "write_atomic(\"a.bin\", [1, 300])").unwrap_err();
    assert!(e.contains("300"), "error should name the culprit: {e}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn non_string_non_array_value_is_a_hard_error() {
    let dir = sandbox("badval");
    let mut env = env_with(&dir);
    // a record is neither a string nor a byte array.
    assert!(eval_value(&mut env, "write_atomic(\"a.bin\", {x: 1})").is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sandbox_escape_is_an_err_result() {
    let dir = sandbox("escape");
    let mut env = env_with(&dir);
    assert_eq!(
        scalar(&mut env, "is_ok(write_atomic(\"../evil.txt\", \"x\"))"),
        0.0
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_sandbox_surface_is_an_err_result() {
    let mut env = Environment::new(); // no fs_root (browser-like)
    assert_eq!(
        scalar(&mut env, "is_ok(write_atomic(\"x.txt\", \"y\"))"),
        0.0
    );
}

#[test]
fn atomic_json_persist_composes() {
    let dir = sandbox("json");
    let mut env = env_with(&dir);
    // durable JSON persist: encode + atomic write, then read + parse.
    eval_value(
        &mut env,
        "write_atomic(\"r.json\", unwrap(to_json({a: 1, b: 2})))",
    )
    .unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal(\"record\", type_of(unwrap(parse_json(unwrap(read_text(\"r.json\"))))))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}
