//! Byte <-> string + parse conversions for the compile-to-Rust path,
//! mirroring the interpreter's `tokenize_bytes` / `decode_bytes` /
//! `to_int`. `tokenize_bytes` yields a plain numeric `DenseArray` (so
//! the bytes flow into arithmetic / reductions); `decode_bytes` and
//! `to_int` yield `CVal`s (a string, and an `ok`/`err` Result).

use mlpl_array::{DenseArray, Shape};

use crate::CVal;

/// `tokenize_bytes(s)` -- a string's UTF-8 bytes as a rank-1 array.
///
/// # Panics
/// Panics if the argument is not a string (interpreter parity: a
/// non-string here is a hard error).
#[must_use]
pub fn tokenize_bytes(v: &CVal) -> DenseArray {
    let CVal::Str(s) = v else {
        panic!("tokenize_bytes: expected a string, got {v:?}");
    };
    let data: Vec<f64> = s.as_bytes().iter().map(|&b| f64::from(b)).collect();
    DenseArray::new(Shape::vector(data.len()), data).expect("byte vector")
}

/// `decode_bytes(bytes)` -- a rank-1 byte array back to a string. Out
/// of `0..=255` / non-integer cells are a hard error (interpreter
/// parity); the bytes are decoded as UTF-8 (lossy for non-UTF-8).
///
/// # Panics
/// Panics on a non-array argument or an invalid byte cell.
#[must_use]
pub fn decode_bytes(v: &CVal) -> CVal {
    let CVal::Arr(a) = v else {
        panic!("decode_bytes: expected an array, got {v:?}");
    };
    let bytes = crate::io::array_to_bytes("decode_bytes", a).unwrap_or_else(|e| panic!("{e}"));
    CVal::Str(String::from_utf8_lossy(&bytes).into_owned())
}

/// `disp(v)` -- format a value to its display string, mirroring the
/// interpreter: an array renders boxed, everything else via its
/// `Display`. Returns a `CVal::Str` (not a side-effecting print) --
/// the program prints the final expression, so `disp(x)` as the last
/// statement shows `x`, exactly as in the interpreter.
#[must_use]
pub fn disp(v: &CVal) -> CVal {
    match v {
        CVal::Arr(a) => CVal::Str(mlpl_array::box_display(a)),
        other => CVal::Str(format!("{other}")),
    }
}

/// `to_int(s)` -- parse a string as an integer -> `ok(int)` / `err`.
///
/// # Panics
/// Panics if the argument is not a string.
#[must_use]
pub fn to_int(v: &CVal) -> CVal {
    let CVal::Str(s) = v else {
        panic!("to_int: expected a string, got {v:?}");
    };
    match s.trim().parse::<i64>() {
        Ok(n) => CVal::result(true, CVal::Arr(DenseArray::from_scalar(n as f64))),
        Err(_) => CVal::result(
            false,
            CVal::Str(format!("to_int: cannot parse {s:?} as an integer")),
        ),
    }
}
