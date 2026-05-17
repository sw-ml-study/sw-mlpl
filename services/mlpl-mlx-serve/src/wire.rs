//! Tensor wire format. Saga R1 step 002.
//!
//! `bincode` 1.x with explicit versioned envelope:
//! `{version: u32 = 1, dtype: u8, ndim: u8, shape: [u64; ndim],
//! data: [u8]}`. Endianness fixed little-endian.
//!
//! Saga 21.5 step 011 expanded the dtype slot from f64-only to
//! a 3-variant tagged union:
//!
//! | Tag | Const          | Wire bytes/elem | Source semantics |
//! |-----|----------------|-----------------|------------------|
//! |  0  | `DTYPE_F64`    | 8 (f64 LE)      | default; what `DenseArray.data()` carries |
//! |  1  | `DTYPE_F32`    | 4 (f32 LE)      | callers that know their tensor fits f32 (training params, activations) |
//! |  2  | `DTYPE_U8`     | 1               | image / mask tensors that materialize as bytes at the source and stay u8 until the first arithmetic op upgrades them |
//!
//! Promotion ladder on the peer: u8 stays u8 only as long as no
//! arithmetic op touches it; the first `add` / `mul` / `matmul`
//! lifts to f32; mixing f32 with an f64 input promotes to f64.
//! Decode ALWAYS lifts back to a `DenseArray` (f64 internally)
//! because that's the orchestrator's representation; the peer
//! does the dtype-specific arithmetic before encoding the
//! reply.
//!
//! `encode_for_json` / `decode_from_json` wrap the bincode
//! bytes in base64 for the JSON-friendly transport used by the
//! eval-on-device endpoint. `encode_tensor` (no `_as`) keeps
//! the default-f64 path byte-for-byte identical to R1 so
//! existing callers see no behavior change.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use mlpl_array::{DenseArray, Shape};
use serde::{Deserialize, Serialize};

const FORMAT_VERSION: u32 = 1;
/// f64 LE (default; what R1 shipped).
pub const DTYPE_F64: u8 = 0;
/// f32 LE; encode casts `DenseArray.data()` (f64) to f32 with
/// f32-precision loss; decode lifts back to f64.
pub const DTYPE_F32: u8 = 1;
/// u8; encode casts via `as u8` (Rust's saturating cast from
/// f64), so values out of [0, 255] clamp rather than wrap.
/// Caller's responsibility to keep image pixels in range to
/// begin with.
pub const DTYPE_U8: u8 = 2;

#[derive(Serialize, Deserialize)]
struct Envelope {
    version: u32,
    dtype: u8,
    ndim: u8,
    shape: Vec<u64>,
    data: Vec<u8>,
}

#[derive(Debug, PartialEq)]
pub enum WireError {
    Serialize(String),
    UnsupportedVersion { saw: u32, expected: u32 },
    UnsupportedDtype(u8),
    NdimShapeMismatch { ndim: u8, shape_len: usize },
    DataLengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialize(s) => write!(f, "wire serialize/deserialize: {s}"),
            Self::UnsupportedVersion { saw, expected } => {
                write!(f, "unsupported wire version {saw}; expected {expected}")
            }
            Self::UnsupportedDtype(d) => write!(
                f,
                "unsupported wire dtype {d}; expected 0 (f64), 1 (f32), or 2 (u8)"
            ),
            Self::NdimShapeMismatch { ndim, shape_len } => write!(
                f,
                "ndim/shape mismatch: ndim={ndim} but shape vec has {shape_len} entries"
            ),
            Self::DataLengthMismatch { expected, actual } => write!(
                f,
                "data length mismatch: expected {expected} bytes, got {actual}"
            ),
        }
    }
}

impl std::error::Error for WireError {}

/// Serialize a `DenseArray` to the default-f64 wire format.
/// Byte-for-byte identical to R1 so existing callers see no
/// behavior change. Saga 21.5 step 011: equivalent to
/// `encode_tensor_as(arr, DTYPE_F64)`.
pub fn encode_tensor(arr: &DenseArray) -> Result<Vec<u8>, WireError> {
    encode_tensor_as(arr, DTYPE_F64)
}

/// Saga 21.5 step 011: serialize a `DenseArray` choosing the
/// on-disk dtype. `DTYPE_F64` writes the values byte-for-byte;
/// `DTYPE_F32` casts via `as f32` (f32-precision loss);
/// `DTYPE_U8` casts via `as u8` (saturating). Unknown dtypes
/// return `UnsupportedDtype`.
pub fn encode_tensor_as(arr: &DenseArray, dtype: u8) -> Result<Vec<u8>, WireError> {
    let dims: Vec<u64> = arr.shape().dims().iter().map(|d| *d as u64).collect();
    let ndim = u8::try_from(dims.len()).map_err(|_| WireError::NdimShapeMismatch {
        ndim: u8::MAX,
        shape_len: dims.len(),
    })?;
    let values = arr.data();
    let data: Vec<u8> = match dtype {
        DTYPE_F64 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DTYPE_F32 => values
            .iter()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect(),
        DTYPE_U8 => values.iter().map(|v| *v as u8).collect(),
        other => return Err(WireError::UnsupportedDtype(other)),
    };
    let env = Envelope {
        version: FORMAT_VERSION,
        dtype,
        ndim,
        shape: dims,
        data,
    };
    bincode::serialize(&env).map_err(|e| WireError::Serialize(e.to_string()))
}

/// Deserialize a tensor envelope back into a `DenseArray`
/// (always f64 internally). Validates version, dtype, and the
/// ndim/shape consistency + data length for the recorded
/// dtype's bytes-per-element.
pub fn decode_tensor(bytes: &[u8]) -> Result<DenseArray, WireError> {
    let env: Envelope =
        bincode::deserialize(bytes).map_err(|e| WireError::Serialize(e.to_string()))?;
    if env.version != FORMAT_VERSION {
        return Err(WireError::UnsupportedVersion {
            saw: env.version,
            expected: FORMAT_VERSION,
        });
    }
    if env.shape.len() != env.ndim as usize {
        return Err(WireError::NdimShapeMismatch {
            ndim: env.ndim,
            shape_len: env.shape.len(),
        });
    }
    let dims: Vec<usize> = env.shape.iter().map(|d| *d as usize).collect();
    let expected_elements: usize = dims.iter().product();
    let floats = decode_data(&env.data, env.dtype, expected_elements)?;
    DenseArray::new(Shape::new(dims), floats)
        .map_err(|e| WireError::Serialize(format!("DenseArray::new: {e}")))
}

/// Inverse of `encode_data`. Validates data-length per dtype's
/// bytes-per-element and lifts every variant back to f64.
fn decode_data(data: &[u8], dtype: u8, expected_elements: usize) -> Result<Vec<f64>, WireError> {
    let (bytes_per_elem, lift): (usize, fn(&[u8]) -> f64) = match dtype {
        DTYPE_F64 => (8, |c| f64::from_le_bytes(c.try_into().unwrap())),
        DTYPE_F32 => (4, |c| f32::from_le_bytes(c.try_into().unwrap()) as f64),
        DTYPE_U8 => (1, |c| c[0] as f64),
        other => return Err(WireError::UnsupportedDtype(other)),
    };
    let expected_bytes = expected_elements * bytes_per_elem;
    if data.len() != expected_bytes {
        return Err(WireError::DataLengthMismatch {
            expected: expected_bytes,
            actual: data.len(),
        });
    }
    Ok(data.chunks_exact(bytes_per_elem).map(lift).collect())
}

/// Encode a tensor and base64-wrap it for the JSON-friendly
/// transport. Default-f64.
pub fn encode_for_json(arr: &DenseArray) -> Result<String, WireError> {
    Ok(BASE64.encode(encode_tensor(arr)?))
}

/// Inverse of `encode_for_json`. Handles all three dtypes
/// because `decode_tensor` does.
pub fn decode_from_json(s: &str) -> Result<DenseArray, WireError> {
    let bytes = BASE64
        .decode(s)
        .map_err(|e| WireError::Serialize(format!("base64: {e}")))?;
    decode_tensor(&bytes)
}
