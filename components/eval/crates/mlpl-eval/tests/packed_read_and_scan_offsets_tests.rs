//! Two memory-lever primitives for ../demo-ml-utils catalog building:
//! `read_bytes_packed` returns a u8-packed `Value::Bytes` (1x memory,
//! not the 8x f64 array), and `scan_length_prefixed_offsets` returns the
//! per-record payload offsets + lengths as arrays (built in Rust) so a
//! caller can store offsets/lengths and lazy-decode names without an
//! MLPL per-element loop.

use std::collections::BTreeMap;

use mlpl_eval::{Environment, Value};

fn eval(env: &mut Environment, src: &str) -> Result<Value, String> {
    let toks = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&toks).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn sandbox(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlpl-packed-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_prefixed(dir: &std::path::Path, name: &str, records: &[&[u8]]) {
    let mut buf = Vec::new();
    for r in records {
        buf.extend_from_slice(&(r.len() as u64).to_le_bytes());
        buf.extend_from_slice(r);
    }
    std::fs::write(dir.join(name), buf).unwrap();
}

fn env_with(dir: &std::path::Path) -> Environment {
    let mut env = Environment::new();
    env.fs_root = Some(dir.to_path_buf());
    env
}

fn ok(env: &mut Environment, src: &str) -> Value {
    match eval(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Result { ok: true, payload } => *payload,
        other => panic!("expected ok(...), got {other:?}"),
    }
}

fn ok_fields(env: &mut Environment, src: &str) -> BTreeMap<String, Value> {
    match ok(env, src) {
        Value::Record { fields } => fields,
        other => panic!("expected ok(record), got {other:?}"),
    }
}

fn vec_field(fields: &BTreeMap<String, Value>, name: &str) -> Vec<f64> {
    match fields.get(name) {
        Some(Value::Array(a)) => a.data().to_vec(),
        other => panic!("field {name}: expected array, got {other:?}"),
    }
}

#[test]
fn read_bytes_packed_returns_u8_bytes() {
    let dir = sandbox("packed");
    std::fs::write(dir.join("d.bin"), [65u8, 66, 67, 68, 69]).unwrap();
    let mut env = env_with(&dir);
    // Whole-file packed read -> a u8 Bytes buffer with the raw bytes.
    match ok(&mut env, "read_bytes_packed(\"d.bin\")") {
        Value::Bytes { data, .. } => assert_eq!(data, vec![65, 66, 67, 68, 69]),
        other => panic!("expected Bytes, got {other:?}"),
    }
    // Range packed read (offset 1, length 3) -> [66, 67, 68].
    match ok(&mut env, "read_bytes_packed(\"d.bin\", 1, 3)") {
        Value::Bytes { data, .. } => assert_eq!(data, vec![66, 67, 68]),
        other => panic!("expected Bytes, got {other:?}"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scan_offsets_collects_per_record_offsets_and_lengths() {
    let dir = sandbox("offsets");
    // "abc"(3), "hello"(5), "hi"(2). Payload offsets: 8, 19, 32.
    write_prefixed(&dir, "g.bin", &[b"abc", b"hello", b"hi"]);
    let mut env = env_with(&dir);
    let f = ok_fields(
        &mut env,
        "scan_length_prefixed_offsets(\"g.bin\", 0, 3, 8, 100, 100, 64)",
    );
    // Payload start offsets (after each 8-byte prefix).
    assert_eq!(vec_field(&f, "offsets"), vec![8.0, 19.0, 32.0]);
    assert_eq!(vec_field(&f, "lengths"), vec![3.0, 5.0, 2.0]);
    // Aggregates still present.
    assert_eq!(
        match f.get("next_offset") {
            Some(Value::Array(a)) => a.data()[0],
            _ => panic!(),
        },
        34.0
    );
    std::fs::remove_dir_all(&dir).ok();
}
