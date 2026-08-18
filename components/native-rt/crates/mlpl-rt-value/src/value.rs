//! The compiled value model. The numerical compile path produces
//! `DenseArray`; `CVal` is the boundary value that also carries
//! strings, string lists, records, and Results, so compiled programs
//! can do string / stdout / args / record / result work. Numerical
//! subexpressions stay `DenseArray` and are wrapped with `CVal::Arr`
//! only where a value crosses into a non-numeric position. The
//! compound constructors (`record`, `result`) live here too.

use std::collections::BTreeMap;
use std::fmt;

use mlpl_array::DenseArray;

/// A compiled program value: a numeric array, a string, a string
/// list (the args vector), a record (sorted fields), or a Result
/// (`ok`/`err` with a payload).
#[derive(Clone, Debug, PartialEq)]
pub enum CVal {
    Arr(DenseArray),
    Str(String),
    StrList(Vec<String>),
    Record(BTreeMap<String, CVal>),
    Result { ok: bool, payload: Box<CVal> },
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

    /// The string payload, for the `str_*` builtins.
    ///
    /// # Panics
    /// Panics if the value is not a string.
    #[must_use]
    pub fn str(&self) -> &str {
        match self {
            CVal::Str(s) => s,
            other => panic!("expected a string, got {other:?}"),
        }
    }

    /// A record field by name.
    ///
    /// # Panics
    /// Panics if the value is not a record, or has no such field.
    #[must_use]
    pub fn field(&self, name: &str) -> &CVal {
        match self {
            CVal::Record(fields) => fields
                .get(name)
                .unwrap_or_else(|| panic!("record has no field `{name}`")),
            other => panic!("expected a record, got {other:?}"),
        }
    }

    /// Build a record from `(name, value)` pairs (sorted by name).
    #[must_use]
    pub fn record(fields: Vec<(String, CVal)>) -> CVal {
        CVal::Record(fields.into_iter().collect::<BTreeMap<_, _>>())
    }

    /// Build an `ok`/`err` Result around a payload.
    #[must_use]
    pub fn result(ok: bool, payload: CVal) -> CVal {
        CVal::Result {
            ok,
            payload: Box::new(payload),
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
            CVal::Record(fields) => {
                let body = fields
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{{{body}}}")
            }
            CVal::Result { ok, payload } => {
                write!(f, "{}({payload})", if *ok { "ok" } else { "err" })
            }
        }
    }
}
