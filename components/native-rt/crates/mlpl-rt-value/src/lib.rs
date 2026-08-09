//! The compiled value model. The numerical compile path produces
//! `DenseArray`; `CVal` is the boundary value that also carries
//! strings and string lists, so compiled programs can do string /
//! stdout / args I/O. Numerical subexpressions stay `DenseArray`
//! and are wrapped with `CVal::Arr` only where a value crosses
//! into a string/IO/result position.

use std::fmt;

use mlpl_array::DenseArray;

mod io;
pub use io::{arg, cli_args, write_stdout};

/// A compiled program value: a numeric array, a string, or a
/// string list (the args vector).
#[derive(Clone, Debug, PartialEq)]
pub enum CVal {
    Arr(DenseArray),
    Str(String),
    StrList(Vec<String>),
}

impl CVal {
    /// The array payload, for numerical call sites that require it
    /// (e.g. the `mlpl!` macro's `result.arr().data()[0]`).
    ///
    /// # Panics
    /// Panics if the value is not an array.
    #[must_use]
    pub fn arr(&self) -> &DenseArray {
        match self {
            CVal::Arr(a) => a,
            other => panic!("expected a numeric array, got {other:?}"),
        }
    }
}

impl From<DenseArray> for CVal {
    fn from(a: DenseArray) -> Self {
        CVal::Arr(a)
    }
}

impl fmt::Display for CVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CVal::Arr(a) => write!(f, "{a}"),
            CVal::Str(s) => write!(f, "{s}"),
            CVal::StrList(items) => write!(f, "{}", items.join("\n")),
        }
    }
}
