//! The `max_elements` decode cap: a cumulative count of the
//! collection elements a parsed value holds -- record fields,
//! array cells, and string-list items, recursively. Enforced
//! after decoding (the pre-parse guard is `max_bytes`); scalars
//! and strings are leaves and count zero.

use mlpl_eval_types::Value;

/// Error if the value's total element count exceeds `max`.
pub(crate) fn check(v: &Value, max: usize) -> Result<(), String> {
    let n = count(v);
    if n > max {
        Err(format!(
            "collection has {n} elements, exceeds max_elements {max}"
        ))
    } else {
        Ok(())
    }
}

fn count(v: &Value) -> usize {
    match v {
        Value::Record { fields } => fields.len() + fields.values().map(count).sum::<usize>(),
        Value::Array(a) if a.rank() >= 1 => a.data().len(),
        Value::StrList { items } => items.len(),
        Value::Result { payload, .. } => count(payload),
        _ => 0,
    }
}
