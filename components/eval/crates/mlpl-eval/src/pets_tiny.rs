//! `load_preloaded("pets_tiny")` -- a 200-image Oxford-IIIT
//! Pet fixture (100 cats + 100 dogs, 64x64 RGB) shipped as a
//! pre-decoded u8 blob and unpacked at REPL call time into a
//! `Value::Record { X, Y, names }`.
//! Saga 29 step 003.
//!
//! Wire format (little-endian throughout):
//!
//! - 8 bytes: magic `MLPLPETS`
//! - 4 bytes: version u32 (currently 1)
//! - 4 bytes: N (count) u32 -- expected 200
//! - 4 bytes: C (channels) u32 -- expected 3
//! - 4 bytes: H (height) u32 -- expected 64
//! - 4 bytes: W (width) u32  -- expected 64
//! - N bytes: Y labels (0 = cat, 1 = dog)
//! - 4 bytes: name-table byte length (u32)
//! - name-table bytes: N filenames separated by `\0`
//! - N * C * H * W bytes: X pixel data, [batch, channel, y, x]
//!   row-major, u8 in [0, 255]
//!
//! `load()` expands u8 -> f64 in [-1, 1] via the same
//! `v / 127.5 - 1.0` normalization the live `load_images`
//! builtin uses, so a model trained on `pets_tiny` and a
//! model trained on the live dataset see the same input
//! distribution.

use std::collections::BTreeMap;

use mlpl_array::{DenseArray, Shape};

use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

const MAGIC: &[u8; 8] = b"MLPLPETS";
const VERSION: u32 = 1;

/// Compiled-in pets_tiny fixture. Built offline by
/// `scripts/build-pets-tiny.rs` from the gitignored
/// `data/oxford-iiit-pet/images/` checkout. Empty until that
/// script runs; the test gate (and any live caller) will
/// fail-fast with a clear message if so.
const PETS_TINY_BIN: &[u8] = include_bytes!("../data/pets_tiny.bin");

/// Read the embedded fixture, expand the u8 pixel data to
/// f64 in [-1, 1], and return a `Value::Record` with fields
/// `X` ([N, 3, H, W] with axis labels), `Y` ([N]), and
/// `names` (a `Value::StrList`).
pub(crate) fn load() -> Result<Value, EvalError> {
    let (header, body) = parse_header(PETS_TINY_BIN)?;
    let Header { n, c, h, w } = header;
    let mut cursor = 0usize;
    if body.len() < n {
        return Err(corrupt("Y labels truncated"));
    }
    let y_bytes = &body[cursor..cursor + n];
    cursor += n;
    if body.len() < cursor + 4 {
        return Err(corrupt("name-table length missing"));
    }
    let names_len = u32::from_le_bytes([
        body[cursor],
        body[cursor + 1],
        body[cursor + 2],
        body[cursor + 3],
    ]) as usize;
    cursor += 4;
    if body.len() < cursor + names_len {
        return Err(corrupt("name-table truncated"));
    }
    let names_bytes = &body[cursor..cursor + names_len];
    cursor += names_len;
    let per_image = c * h * w;
    let pixel_bytes = n * per_image;
    if body.len() < cursor + pixel_bytes {
        return Err(corrupt("pixel data truncated"));
    }
    let pixels = &body[cursor..cursor + pixel_bytes];

    let names: Vec<String> = if names_len == 0 {
        Vec::new()
    } else {
        names_bytes
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect()
    };

    // X: u8 -> f64 in [-1, 1]
    let mut x_data = Vec::with_capacity(pixel_bytes);
    for &b in pixels {
        x_data.push(f64::from(b) / 127.5 - 1.0);
    }
    let x_arr = DenseArray::new(Shape::new(vec![n, c, h, w]), x_data)?.with_labels(vec![
        Some("batch".to_string()),
        Some("channel".to_string()),
        Some("y".to_string()),
        Some("x".to_string()),
    ])?;

    let y_data: Vec<f64> = y_bytes.iter().map(|b| f64::from(*b)).collect();
    let y_arr = DenseArray::new(Shape::new(vec![n]), y_data)?
        .with_labels(vec![Some("batch".to_string())])?;

    let mut fields = BTreeMap::new();
    fields.insert("X".to_string(), Value::Array(x_arr));
    fields.insert("Y".to_string(), Value::Array(y_arr));
    fields.insert("names".to_string(), Value::StrList { items: names });
    Ok(Value::Record { fields })
}

struct Header {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

fn parse_header(bytes: &[u8]) -> Result<(Header, &[u8]), EvalError> {
    if bytes.is_empty() {
        return Err(EvalError::Unsupported(
            "load_preloaded(\"pets_tiny\"): fixture is empty -- run \
             scripts/build-pets-tiny.rs against the gitignored \
             data/oxford-iiit-pet/ checkout to populate it"
                .into(),
        ));
    }
    if bytes.len() < 8 + 4 + 16 {
        return Err(corrupt("file shorter than header"));
    }
    if &bytes[..8] != MAGIC {
        return Err(corrupt("bad magic bytes"));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != VERSION {
        return Err(EvalError::Unsupported(format!(
            "load_preloaded(\"pets_tiny\"): version {version} not supported (this build expects {VERSION})"
        )));
    }
    let read_u32 = |o: usize| {
        u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
    };
    let n = read_u32(12);
    let c = read_u32(16);
    let h = read_u32(20);
    let w = read_u32(24);
    Ok((Header { n, c, h, w }, &bytes[28..]))
}

fn corrupt(msg: &str) -> EvalError {
    EvalError::Unsupported(format!(
        "load_preloaded(\"pets_tiny\"): corrupt fixture ({msg})"
    ))
}
