//! Registry-row and annotation-record construction for the
//! reflection builtins (`test_info` / `annotations`).

use std::collections::BTreeMap;

use mlpl_array::DenseArray;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_state::TestEntry;
use mlpl_eval_types::{EvalError, Value};

/// `test_info("stable name")` -> the registry row as a record;
/// `fn` is the `:u:` reference a runner can `call`.
pub(crate) fn test_info(key: &str, env: &Environment) -> Result<Value, EvalError> {
    let Some(t) = env.tests.iter().find(|t| t.name == key) else {
        let known: Vec<String> = env.tests.iter().map(|t| t.name.clone()).collect();
        return Err(EvalError::Unsupported(format!(
            "test_info: no test named \"{key}\" (registered: {})",
            known.join(", ")
        )));
    };
    Ok(Value::Record {
        fields: registry_row(t),
    })
}

/// The registry row's record fields; `fn` is the callable ref.
#[allow(clippy::cast_precision_loss)]
fn registry_row(t: &TestEntry) -> BTreeMap<String, Value> {
    let scalar = |v: f64| Value::Array(DenseArray::from_scalar(v));
    BTreeMap::from([
        ("name".into(), Value::Str(t.name.clone())),
        (
            "fn".into(),
            Value::UserFnRef {
                name: t.fn_name.clone(),
            },
        ),
        (
            "tags".into(),
            Value::StrList {
                items: t.tags.clone(),
            },
        ),
        ("skip".into(), Value::Str(t.skip.clone())),
        ("expected_failure".into(), scalar(t.expected_failure)),
        ("timeout_ms".into(), scalar(t.timeout_ms)),
        ("source".into(), Value::Str(t.source.clone())),
        ("line".into(), scalar(t.line as f64)),
    ])
}

/// `annotations("u:name")` -> `{word: payload, ...}`; bare
/// annotations map to scalar 1. The general-namespace reader
/// (documentation/math extraction).
pub(crate) fn annotations_of(
    key: &str,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let full = if key.contains(':') {
        key.to_string()
    } else {
        format!("u:{key}")
    };
    let anns = env
        .get_fn(&full)
        .ok_or_else(|| EvalError::Unsupported(format!("undefined function: {full}")))?
        .annotations
        .clone();
    let mut fields = BTreeMap::new();
    for (word, payload) in anns {
        let value = match payload {
            Some(expr) => crate::eval::eval_expr(&expr, env, trace)?,
            None => Value::Array(DenseArray::from_scalar(1.0)),
        };
        fields.insert(word, value);
    }
    Ok(Value::Record { fields })
}
