//! Envelope validation for `emit_test_event` (schema version 1):
//! required common fields, known kinds, end-status enum. Unknown
//! ADDITIVE fields are deliberately ignored -- consumers of the
//! same major version must tolerate them, so the emitter does.

use std::collections::BTreeMap;

use crate::fncall_events::EVENT_KINDS;
use mlpl_eval_types::{EvalError, Value};

const END_STATUSES: &[&str] = &["passed", "failed", "skipped", "expected_failure"];

pub(crate) fn validate(fields: &BTreeMap<String, Value>) -> Result<(), EvalError> {
    match fields.get("version") {
        Some(Value::Array(a)) if a.data() == [1.0] => {}
        other => {
            return Err(bad(format!(
                "`version` must be the scalar 1 (schema version) -- got {}",
                describe(other)
            )));
        }
    }
    let kind = required_str(fields, "kind")?;
    if !EVENT_KINDS.contains(&kind) {
        return Err(bad(format!(
            "unknown kind `{kind}` -- known kinds: {}",
            EVENT_KINDS.join(", ")
        )));
    }
    required_str(fields, "suite")?;
    required_str(fields, "name")?;
    if kind.ends_with("_end") && kind != "suite_end" {
        let status = required_str(fields, "status")?;
        if !END_STATUSES.contains(&status) {
            return Err(bad(format!(
                "`status` must be one of {} -- got `{status}`",
                END_STATUSES.join(", ")
            )));
        }
    }
    Ok(())
}

fn required_str<'a>(fields: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a str, EvalError> {
    match fields.get(key) {
        Some(Value::Str(s)) => Ok(s),
        other => Err(bad(format!(
            "`{key}` is required as a string -- got {}",
            describe(other)
        ))),
    }
}

fn describe(v: Option<&Value>) -> String {
    v.map_or_else(
        || "nothing".to_string(),
        |v| mlpl_eval_types::value_kind(v).to_string(),
    )
}

fn bad(reason: String) -> EvalError {
    EvalError::Unsupported(format!("emit_test_event: {reason}"))
}
