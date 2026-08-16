//! `file_metadata(path)` -- the sandbox-confined last-MODIFIED-time
//! primitive (../demo-extensions model picker + ../demo-file-processing
//! date demos). Returns `ok({kind, size, modified_unix_ms})` / `err`,
//! Unix-ms integer, err when unavailable, same sandbox + symlink rules
//! as `file_size`. The timestamp tests pin a KNOWN mtime so they never
//! depend on the wall clock -- which also proves the value is the
//! modification time, not the current clock.

use std::collections::BTreeMap;
use std::time::{Duration, UNIX_EPOCH};

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn sandbox(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlpl-meta-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn env_with(dir: &std::path::Path) -> Environment {
    let mut env = Environment::new();
    env.fs_root = Some(dir.to_path_buf());
    env
}

/// Write `rel` under `dir` and stamp its mtime to `ms` since the epoch.
fn write_with_mtime(dir: &std::path::Path, rel: &str, contents: &[u8], ms: u64) {
    let path = dir.join(rel);
    std::fs::write(&path, contents).unwrap();
    let f = std::fs::File::options().write(true).open(&path).unwrap();
    f.set_modified(UNIX_EPOCH + Duration::from_millis(ms))
        .unwrap();
}

/// Eval a `file_metadata` call and return the `ok` record's fields.
fn ok_fields(env: &mut Environment, src: &str) -> BTreeMap<String, Value> {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Result { ok: true, payload } => match *payload {
            Value::Record { fields } => fields,
            other => panic!("expected a record payload, got {other:?}"),
        },
        other => panic!("expected ok(record), got {other:?}"),
    }
}

fn scalar_field(fields: &BTreeMap<String, Value>, name: &str) -> f64 {
    match fields.get(name) {
        Some(Value::Array(a)) => a.data()[0],
        other => panic!("field {name}: expected scalar, got {other:?}"),
    }
}

fn is_err(env: &mut Environment, src: &str) -> bool {
    matches!(
        eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")),
        Value::Result { ok: false, .. }
    )
}

#[test]
fn file_metadata_reports_kind_size_and_exact_modified_ms() {
    let dir = sandbox("file");
    let mut env = env_with(&dir);
    // A known mtime in the past (Sept 2020) -- NOT the current clock.
    let known_ms: u64 = 1_600_000_000_123;
    write_with_mtime(&dir, "m.bin", &[1, 2, 3, 4, 5], known_ms);

    let fields = ok_fields(&mut env, "file_metadata(\"m.bin\")");
    assert_eq!(fields.get("kind"), Some(&Value::Str("file".into())));
    assert_eq!(scalar_field(&fields, "size"), 5.0);
    // Exact Unix-millisecond modification time (proves it is mtime, not
    // now / atime / ctime, and that ms precision is carried).
    assert_eq!(scalar_field(&fields, "modified_unix_ms"), known_ms as f64);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn file_metadata_reports_directory_kind() {
    let dir = sandbox("dir");
    let mut env = env_with(&dir);
    std::fs::create_dir(dir.join("sub")).unwrap();
    let fields = ok_fields(&mut env, "file_metadata(\"sub\")");
    assert_eq!(fields.get("kind"), Some(&Value::Str("dir".into())));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn file_metadata_missing_file_is_err() {
    let dir = sandbox("missing");
    let mut env = env_with(&dir);
    assert!(is_err(&mut env, "file_metadata(\"nope.bin\")"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn file_metadata_without_sandbox_is_err() {
    let mut env = Environment::new(); // no fs_root configured
    assert!(is_err(&mut env, "file_metadata(\"any.bin\")"));
}

#[cfg(unix)]
#[test]
fn file_metadata_refuses_symlink_escape() {
    let dir = sandbox("escape");
    let mut env = env_with(&dir);
    // A secret outside the sandbox, and a symlink to it inside.
    let outside = std::env::temp_dir().join(format!("mlpl-meta-{}-secret", std::process::id()));
    std::fs::write(&outside, b"secret").unwrap();
    std::os::unix::fs::symlink(&outside, dir.join("link.bin")).unwrap();
    assert!(
        is_err(&mut env, "file_metadata(\"link.bin\")"),
        "a symlink escaping the sandbox must be refused"
    );
    std::fs::remove_file(&outside).ok();
    std::fs::remove_dir_all(&dir).ok();
}
