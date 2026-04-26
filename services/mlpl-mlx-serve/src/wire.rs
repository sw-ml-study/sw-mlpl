//! Tensor wire format. Saga R1 step 002.
//!
//! `bincode` 1.x with explicit versioned envelope:
//! `{version: u32 = 1, dtype: u8 = 0 (f64), ndim:
//! u8, shape: [u64; ndim], data: [u8]}`. Endianness
//! fixed little-endian. The version field reserves
//! space for future dtype expansion (f32 / bf16 /
//! f16 ship in a separate dtype saga).
//!
//! `encode_for_json` / `decode_from_json` wrap the
//! bincode bytes in base64 for the JSON-friendly
//! transport used by the eval-on-device endpoint.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use mlpl_array::{DenseArray, Shape};
use serde::{Deserialize, Serialize};

const FORMAT_VERSION: u32 = 1;
const DTYPE_F64: u8 = 0;

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
            Self::UnsupportedDtype(d) => {
                write!(f, "unsupported wire dtype {d}; R1 supports only 0 (f64)")
            }
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

/// Serialize a `DenseArray` to the bincode +
/// versioned envelope. f64-only in R1.
pub fn encode_tensor(arr: &DenseArray) -> Result<Vec<u8>, WireError> {
    let dims: Vec<u64> = arr.shape().dims().iter().map(|d| *d as u64).collect();
    let ndim = u8::try_from(dims.len()).map_err(|_| WireError::NdimShapeMismatch {
        ndim: u8::MAX,
        shape_len: dims.len(),
    })?;
    let mut data = Vec::with_capacity(arr.data().len() * 8);
    for v in arr.data() {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let env = Envelope {
        version: FORMAT_VERSION,
        dtype: DTYPE_F64,
        ndim,
        shape: dims,
        data,
    };
    bincode::serialize(&env).map_err(|e| WireError::Serialize(e.to_string()))
}

/// Deserialize a tensor envelope back into a
/// `DenseArray`. Validates version + dtype + the
/// ndim/shape consistency + data length.
pub fn decode_tensor(bytes: &[u8]) -> Result<DenseArray, WireError> {
    let env: Envelope =
        bincode::deserialize(bytes).map_err(|e| WireError::Serialize(e.to_string()))?;
    if env.version != FORMAT_VERSION {
        return Err(WireError::UnsupportedVersion {
            saw: env.version,
            expected: FORMAT_VERSION,
        });
    }
    if env.dtype != DTYPE_F64 {
        return Err(WireError::UnsupportedDtype(env.dtype));
    }
    if env.shape.len() != env.ndim as usize {
        return Err(WireError::NdimShapeMismatch {
            ndim: env.ndim,
            shape_len: env.shape.len(),
        });
    }
    let dims: Vec<usize> = env.shape.iter().map(|d| *d as usize).collect();
    let expected_elements: usize = dims.iter().product();
    let expected_bytes = expected_elements * 8;
    if env.data.len() != expected_bytes {
        return Err(WireError::DataLengthMismatch {
            expected: expected_bytes,
            actual: env.data.len(),
        });
    }
    let mut floats = Vec::with_capacity(expected_elements);
    for chunk in env.data.chunks_exact(8) {
        floats.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    DenseArray::new(Shape::new(dims), floats)
        .map_err(|e| WireError::Serialize(format!("DenseArray::new: {e}")))
}

/// Encode a tensor and base64-wrap it for the
/// JSON-friendly transport.
pub fn encode_for_json(arr: &DenseArray) -> Result<String, WireError> {
    Ok(BASE64.encode(encode_tensor(arr)?))
}

/// Inverse of `encode_for_json`.
pub fn decode_from_json(s: &str) -> Result<DenseArray, WireError> {
    let bytes = BASE64
        .decode(s)
        .map_err(|e| WireError::Serialize(format!("base64: {e}")))?;
    decode_tensor(&bytes)
}
