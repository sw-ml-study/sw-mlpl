//! `CVal` constructors for the compound variants -- records and
//! Results. Kept out of `value.rs` so neither module exceeds the
//! function-count budget.

use std::collections::BTreeMap;

use crate::CVal;

impl CVal {
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
