//! `scan_length_prefixed(path, offset, count, length_width,
//! max_item_bytes, max_total_bytes, chunk_bytes)` -- a bounded-memory
//! streaming scan over `count` little-endian length-prefixed records
//! (../demo-ml-utils GGUF array streaming). It reads each
//! `length_width`-byte prefix and seeks over the payload, retaining no
//! payload bytes, and returns a scalar-record aggregate:
//! `ok({next_offset, item_count, payload_bytes, bytes_read,
//! max_item_seen})`, or an `err` on a bound violation / truncation /
//! sandbox failure.

use std::collections::BTreeMap;

use mlpl_eval::{Environment, Value};

fn eval(env: &mut Environment, src: &str) -> Result<Value, String> {
    let toks = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&toks).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn sandbox(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlpl-scan-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write `records` as little-endian u64-length-prefixed payloads.
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

/// Eval a scan call and return the `ok` record's fields.
fn ok_fields(env: &mut Environment, src: &str) -> BTreeMap<String, Value> {
    match eval(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Result { ok: true, payload } => match *payload {
            Value::Record { fields } => fields,
            other => panic!("expected ok(record), got {other:?}"),
        },
        other => panic!("expected ok(record), got {other:?}"),
    }
}

fn scalar(fields: &BTreeMap<String, Value>, name: &str) -> f64 {
    match fields.get(name) {
        Some(Value::Array(a)) => a.data()[0],
        other => panic!("field {name}: expected scalar, got {other:?}"),
    }
}

fn is_err(env: &mut Environment, src: &str) -> bool {
    matches!(
        eval(env, src).unwrap_or_else(|e| panic!("{src}: {e}")),
        Value::Result { ok: false, .. }
    )
}

#[test]
fn scan_folds_length_prefixed_records() {
    let dir = sandbox("fold");
    // 3 records: "abc"(3), "hello"(5), "hi"(2). Each = 8-byte LE prefix
    // + payload. Total = (8+3)+(8+5)+(8+2) = 34 bytes.
    write_prefixed(&dir, "g.bin", &[b"abc", b"hello", b"hi"]);
    let mut env = env_with(&dir);
    let f = ok_fields(
        &mut env,
        "scan_length_prefixed(\"g.bin\", 0, 3, 8, 100, 100, 64)",
    );
    assert_eq!(scalar(&f, "item_count"), 3.0);
    assert_eq!(scalar(&f, "payload_bytes"), 10.0); // 3 + 5 + 2
    assert_eq!(scalar(&f, "bytes_read"), 34.0); // 24 prefix + 10 payload
    assert_eq!(scalar(&f, "next_offset"), 34.0);
    assert_eq!(scalar(&f, "max_item_seen"), 5.0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scan_from_a_nonzero_offset_and_partial_count() {
    let dir = sandbox("offset");
    write_prefixed(&dir, "g.bin", &[b"abc", b"hello", b"hi"]);
    let mut env = env_with(&dir);
    // Start at the SECOND record (offset 11 = 8+3) and scan 1 item.
    let f = ok_fields(
        &mut env,
        "scan_length_prefixed(\"g.bin\", 11, 1, 8, 100, 100, 64)",
    );
    assert_eq!(scalar(&f, "item_count"), 1.0);
    assert_eq!(scalar(&f, "payload_bytes"), 5.0);
    assert_eq!(scalar(&f, "next_offset"), 24.0); // 11 + 8 + 5
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scan_rejects_oversized_item_and_total() {
    let dir = sandbox("bounds");
    write_prefixed(&dir, "g.bin", &[b"abc", b"hello", b"hi"]);
    let mut env = env_with(&dir);
    // max_item_bytes = 4 rejects the 5-byte "hello".
    assert!(is_err(
        &mut env,
        "scan_length_prefixed(\"g.bin\", 0, 3, 8, 4, 100, 64)"
    ));
    // max_total_bytes = 6 rejects once the running payload exceeds it.
    assert!(is_err(
        &mut env,
        "scan_length_prefixed(\"g.bin\", 0, 3, 8, 100, 6, 64)"
    ));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scan_errs_on_truncation_and_no_sandbox() {
    let dir = sandbox("trunc");
    write_prefixed(&dir, "g.bin", &[b"abc"]);
    let mut env = env_with(&dir);
    // Only 1 record exists; asking for 3 hits EOF -> err.
    assert!(is_err(
        &mut env,
        "scan_length_prefixed(\"g.bin\", 0, 3, 8, 100, 100, 64)"
    ));
    // No sandbox configured -> err.
    let mut bare = Environment::new();
    assert!(is_err(
        &mut bare,
        "scan_length_prefixed(\"g.bin\", 0, 1, 8, 100, 100, 64)"
    ));
    std::fs::remove_dir_all(&dir).ok();
}
