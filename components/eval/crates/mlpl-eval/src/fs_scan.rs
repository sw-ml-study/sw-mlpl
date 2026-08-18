//! `scan_length_prefixed(path, offset, count, length_width,
//! max_item_bytes, max_total_bytes, chunk_bytes)` -- a bounded-memory
//! streaming scan over `count` little-endian length-prefixed records
//! (../demo-ml-utils GGUF array streaming). It reads each
//! `length_width`-byte prefix and SEEKS over the payload, retaining no
//! payload bytes (O(chunk_bytes) buffer, constant native stack), and
//! folds the records into a scalar-record aggregate. The `_offsets`
//! variant ALSO returns the per-record payload offset + length arrays
//! (built in Rust), so a caller can store offsets/lengths and
//! lazy-decode names without an MLPL per-element loop. Sandboxed like
//! `read_bytes`; every bound violation / truncation is a clean `err`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok};
use mlpl_eval_types::{EvalError, Value};

/// `scan_length_prefixed(...)` -> `ok({next_offset, item_count,
/// payload_bytes, bytes_read, max_item_seen})` / `err`.
pub(crate) fn eval_scan(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    dispatch(args, env, trace, false)
}

/// `scan_length_prefixed_offsets(...)` -- the same, plus `offsets` and
/// `lengths` arrays (the per-record payload start offset + byte length).
pub(crate) fn eval_scan_offsets(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    dispatch(args, env, trace, true)
}

/// Shared shell: validate the 7 args (a path + six non-negative integer
/// scalars), resolve the sandbox root, and run the scan.
fn dispatch(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    collect: bool,
) -> Result<Value, EvalError> {
    if args.len() != 7 {
        return Err(EvalError::BadArity {
            func: "scan_length_prefixed".into(),
            expected: 7,
            got: args.len(),
        });
    }
    let Value::Str(rel) = crate::eval::eval_expr(&args[0], env, trace)? else {
        return Err(EvalError::Unsupported(
            "scan_length_prefixed: the first argument is a path string".into(),
        ));
    };
    let mut n = [0u64; 6];
    for (i, a) in args[1..].iter().enumerate() {
        n[i] = nonneg_int(&crate::eval::eval_expr(a, env, trace)?)?;
    }
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(
            "scan_length_prefixed: no filesystem sandbox on this surface".into(),
        ));
    };
    Ok(match scan(&root, &rel, n, collect) {
        Ok(record) => fs_ok(record),
        Err(e) => fs_err(format!("scan_length_prefixed: {e}")),
    })
}

/// Walk the records, reading only prefixes and seeking over payloads.
/// `n` is `[offset, count, width, max_item, max_total, chunk]`. With
/// `collect`, also gather each payload's start offset + length.
fn scan(root: &Path, rel: &str, n: [u64; 6], collect: bool) -> Result<Value, String> {
    let [offset, count, width, max_item, max_total, chunk] = n;
    if width == 0 || width > 8 {
        return Err(format!("length_width must be 1..=8, got {width}"));
    }
    let file = std::fs::File::open(contained(root, rel)?).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut reader = BufReader::with_capacity(chunk.max(1) as usize, file);
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    let (mut cursor, mut payload, mut bytes, mut max_seen) = (offset, 0u64, 0u64, 0u64);
    let (mut offs, mut lens) = (Vec::new(), Vec::new());
    for _ in 0..count {
        let item = scan_item(&mut reader, width, cursor, size, max_item)?;
        payload += item;
        if payload > max_total {
            return Err(format!(
                "payload {payload} exceeds max_total_bytes {max_total}"
            ));
        }
        if collect {
            offs.push((cursor + width) as f64);
            lens.push(item as f64);
        }
        (bytes, cursor, max_seen) = (
            bytes + width + item,
            cursor + width + item,
            max_seen.max(item),
        );
    }
    Ok(record(
        cursor,
        count,
        payload,
        bytes,
        max_seen,
        collect.then_some((offs, lens)),
    ))
}

/// Read one record's length prefix, validate it (bounds + max item),
/// and seek over the payload. Returns the item's byte length.
fn scan_item(
    reader: &mut BufReader<File>,
    width: u64,
    cursor: u64,
    size: u64,
    max_item: u64,
) -> Result<u64, String> {
    if cursor + width > size {
        return Err("truncated length prefix".into());
    }
    let mut prefix = [0u8; 8];
    reader
        .read_exact(&mut prefix[..width as usize])
        .map_err(|e| e.to_string())?;
    let item = prefix[..width as usize]
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, &b)| acc | (u64::from(b) << (8 * i)));
    if item > max_item {
        return Err(format!(
            "item length {item} exceeds max_item_bytes {max_item}"
        ));
    }
    if cursor + width + item > size {
        return Err("truncated payload".into());
    }
    reader
        .seek_relative(item as i64)
        .map_err(|e| e.to_string())?;
    Ok(item)
}

/// A scalar non-negative integer argument, or a hard error.
fn nonneg_int(v: &Value) -> Result<u64, EvalError> {
    match v {
        Value::Array(a) if a.rank() == 0 && a.data()[0] >= 0.0 && a.data()[0].fract() == 0.0 => {
            Ok(a.data()[0] as u64)
        }
        _ => Err(EvalError::Unsupported(
            "scan_length_prefixed: offset/count/widths/budgets must be non-negative integers"
                .into(),
        )),
    }
}

/// Assemble the aggregate record, plus `offsets`/`lengths` when
/// collected. Every scalar field is a rank-0 cell.
fn record(
    next_offset: u64,
    item_count: u64,
    payload_bytes: u64,
    bytes_read: u64,
    max_item: u64,
    collected: Option<(Vec<f64>, Vec<f64>)>,
) -> Value {
    let cell = |v: u64| Value::Array(DenseArray::from_scalar(v as f64));
    let mut fields = BTreeMap::new();
    fields.insert("next_offset".into(), cell(next_offset));
    fields.insert("item_count".into(), cell(item_count));
    fields.insert("payload_bytes".into(), cell(payload_bytes));
    fields.insert("bytes_read".into(), cell(bytes_read));
    fields.insert("max_item_seen".into(), cell(max_item));
    if let Some((offs, lens)) = collected {
        fields.insert("offsets".into(), Value::Array(DenseArray::from_vec(offs)));
        fields.insert("lengths".into(), Value::Array(DenseArray::from_vec(lens)));
    }
    Value::Record { fields }
}
