//! Exception-free record access (demo-functional-pipelines
//! request): `has_field(record, name)` -> 0/1 and
//! `record_get(record, name)` -> `ok(value)` / `err({kind,
//! field, message, available})`, so schema validation is data,
//! not a caught hard error.

use std::collections::BTreeMap;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "has_field" => Some(has_field(args, env, trace)),
        "record_get" => Some(record_get(args, env, trace)),
        _ => None,
    }
}

/// Evaluate the `(record, name)` argument pair, both type-checked.
fn record_and_name(
    who: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(BTreeMap<String, Value>, String), EvalError> {
    let [rec_arg, name_arg] = args else {
        return Err(EvalError::BadArity {
            func: who.into(),
            expected: 2,
            got: args.len(),
        });
    };
    let Value::Record { fields } = crate::eval::eval_expr(rec_arg, env, trace)? else {
        return Err(EvalError::Unsupported(format!(
            "{who}: the first argument must be a record"
        )));
    };
    let Value::Str(name) = crate::eval::eval_expr(name_arg, env, trace)? else {
        return Err(EvalError::Unsupported(format!(
            "{who}: the field name must be a string"
        )));
    };
    Ok((fields, name))
}

fn has_field(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (fields, name) = record_and_name("has_field", args, env, trace)?;
    Ok(Value::Array(DenseArray::from_scalar(f64::from(u8::from(
        fields.contains_key(&name),
    )))))
}

fn record_get(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (fields, name) = record_and_name("record_get", args, env, trace)?;
    match fields.get(&name) {
        Some(v) => Ok(Value::Result {
            ok: true,
            payload: Box::new(v.clone()),
        }),
        None => {
            let available = fields.keys().cloned().collect::<Vec<_>>().join(", ");
            let err = BTreeMap::from([
                ("kind".to_string(), Value::Str("missing_field".to_string())),
                ("field".to_string(), Value::Str(name.clone())),
                (
                    "message".to_string(),
                    Value::Str(format!("no field `{name}` (present: {available})")),
                ),
                ("available".to_string(), Value::Str(available)),
            ]);
            Ok(Value::Result {
                ok: false,
                payload: Box::new(Value::Record { fields: err }),
            })
        }
    }
}
