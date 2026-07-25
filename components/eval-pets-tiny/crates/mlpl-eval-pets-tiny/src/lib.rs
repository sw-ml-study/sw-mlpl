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

pub(crate) const MAGIC: &[u8; 8] = b"MLPLPETS";
pub(crate) const VERSION: u32 = 1;

mod parse;
use parse::{Header, Sections};

/// Compiled-in pets_tiny fixture. Built offline by
/// `scripts/build-pets-tiny.rs` from the gitignored
/// `data/oxford-iiit-pet/images/` checkout. Empty until that
/// script runs; the test gate (and any live caller) will
/// fail-fast with a clear message if so.
const PETS_TINY_BIN: &[u8] = include_bytes!("../../../../eval/crates/mlpl-eval/data/pets_tiny.bin");

/// Read the embedded fixture, expand the u8 pixel data to
/// f64 in [-1, 1], and return a `Value::Record` with fields
/// `X` ([N, 3, H, W] with axis labels), `Y` ([N]), and
/// `names` (a `Value::StrList`).
pub fn load() -> Result<Value, EvalError> {
    let (header, body) = parse::parse_header(PETS_TINY_BIN)?;
    let Sections {
        y_bytes,
        names_bytes,
        pixels,
    } = parse::parse_sections(body, &header)?;
    let names = decode_names(names_bytes);
    let x_arr = build_x_array(pixels, &header)?;
    let y_arr = build_y_array(y_bytes)?;
    let mut fields = BTreeMap::new();
    fields.insert("X".to_string(), Value::Array(x_arr));
    fields.insert("Y".to_string(), Value::Array(y_arr));
    fields.insert("names".to_string(), Value::StrList { items: names });
    Ok(Value::Record { fields })
}

fn decode_names(names_bytes: &[u8]) -> Vec<String> {
    if names_bytes.is_empty() {
        return Vec::new();
    }
    names_bytes
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect()
}

fn build_x_array(pixels: &[u8], h: &Header) -> Result<DenseArray, EvalError> {
    // X: u8 -> f64 in [-1, 1]
    let x_data: Vec<f64> = pixels.iter().map(|b| f64::from(*b) / 127.5 - 1.0).collect();
    Ok(
        DenseArray::new(Shape::new(vec![h.n, h.c, h.h, h.w]), x_data)?.with_labels(vec![
            Some("batch".to_string()),
            Some("channel".to_string()),
            Some("y".to_string()),
            Some("x".to_string()),
        ])?,
    )
}

fn build_y_array(y_bytes: &[u8]) -> Result<DenseArray, EvalError> {
    let y_data: Vec<f64> = y_bytes.iter().map(|b| f64::from(*b)).collect();
    Ok(DenseArray::new(Shape::new(vec![y_data.len()]), y_data)?
        .with_labels(vec![Some("batch".to_string())])?)
}

pub(crate) fn corrupt(msg: &str) -> EvalError {
    EvalError::Unsupported(format!(
        "load_preloaded(\"pets_tiny\"): corrupt fixture ({msg})"
    ))
}
