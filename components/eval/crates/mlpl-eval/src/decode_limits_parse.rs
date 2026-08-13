//! Parse a decode-`Limits` options record: read the optional
//! `max_depth` / `max_bytes` / `max_elements` / `results` fields.
//! Split out of `decode_limits` (per docs/code_metrics.md, split by
//! responsibility) -- the option-record parsing, separate from the
//! codec-facing `text_and_options` / `limits_only` entry points.

use std::collections::BTreeMap;

use crate::decode_limits::Limits;
use mlpl_eval_types::{EvalError, Value};

/// Resolve `Limits` (and the `results` reconstruct flag) from an
/// already-evaluated optional options record.
pub(crate) fn from_option(who: &str, opt: Option<&Value>) -> Result<(Limits, bool), EvalError> {
    let mut limits = Limits::defaults();
    let Some(v) = opt else {
        return Ok((limits, false));
    };
    let Value::Record { fields } = v else {
        return Err(EvalError::Unsupported(format!(
            "{who}: the options argument must be a record"
        )));
    };
    if let Some(d) = usize_field(who, fields, "max_depth")? {
        limits.max_depth = d;
    }
    if let Some(b) = usize_field(who, fields, "max_bytes")? {
        limits.max_bytes = b;
    }
    if let Some(e) = usize_field(who, fields, "max_elements")? {
        limits.max_elements = e;
    }
    let reconstruct = usize_field(who, fields, "results")?.is_some_and(|n| n != 0);
    Ok((limits, reconstruct))
}

fn usize_field(
    who: &str,
    fields: &BTreeMap<String, Value>,
    key: &str,
) -> Result<Option<usize>, EvalError> {
    match fields.get(key) {
        None => Ok(None),
        Some(Value::Array(a))
            if a.rank() == 0 && a.data()[0] >= 0.0 && a.data()[0].fract() == 0.0 =>
        {
            Ok(Some(a.data()[0] as usize))
        }
        Some(_) => Err(EvalError::Unsupported(format!(
            "{who}: {key} must be a non-negative integer"
        ))),
    }
}
