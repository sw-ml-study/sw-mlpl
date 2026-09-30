//! Record reads: field access `r.field` and destructuring
//! `{a, b: x} = value`. Both share one field check, so a non-record or a
//! missing field reports the same error either way. Destructuring
//! evaluates the value once and checks every requested field BEFORE
//! anything binds, so a missing field leaves the scope untouched. Each
//! binding then runs as an ordinary assignment `var = <record>.field` --
//! same kind dispatch, auto-tags, and undo-log frame scoping as
//! `var = expr` -- over a hidden temporary no source identifier can name.

use std::collections::BTreeMap;

use mlpl_core::Span;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::env_api::*;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value, value_kind};

/// Not lexable as an identifier, so it can never shadow a user name.
const TEMP: &str = "{destructure}";

/// `Some(result)` for a field access or a destructuring assignment, `None`
/// for every other form.
pub(crate) fn try_record_form(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match expr {
        Expr::FieldAccess {
            receiver, field, ..
        } => Some(eval_expr(receiver, env, trace).and_then(|recv| {
            let pattern = [(field.clone(), field.clone())];
            let mut fields = checked_fields(&pattern, recv)?;
            Ok(fields
                .remove(field)
                .expect("checked_fields verified the field"))
        })),
        Expr::Destructure {
            bindings, value, ..
        } => Some(eval_destructure(bindings, value, env, trace)),
        _ => None,
    }
}

/// Evaluate `{bindings} = value`; the statement's value is the record.
fn eval_destructure(
    bindings: &[(String, String)],
    value: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let fields = checked_fields(bindings, eval_expr(value, env, trace)?)?;
    env.set_record(TEMP.to_string(), fields.clone());
    let bound = bindings
        .iter()
        .try_for_each(|(field, var)| bind_field(field, var, value.span(), env, trace));
    env.clear_binding(TEMP);
    bound.map(|()| Value::Record { fields })
}

/// The record's fields, once every field the pattern names is present.
fn checked_fields(
    bindings: &[(String, String)],
    v: Value,
) -> Result<BTreeMap<String, Value>, EvalError> {
    let Value::Record { fields } = v else {
        return Err(EvalError::FieldOnNonRecord {
            receiver_kind: value_kind(&v),
            field: bindings[0].0.clone(),
        });
    };
    match bindings.iter().find(|(f, _)| !fields.contains_key(f)) {
        Some((missing, _)) => Err(EvalError::FieldNotFound {
            requested: missing.clone(),
            available: fields.keys().cloned().collect(),
        }),
        None => Ok(fields),
    }
}

/// `var = {destructure}.field`, through the normal assignment path.
fn bind_field(
    field: &str,
    var: &str,
    span: Span,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(), EvalError> {
    let read = Expr::FieldAccess {
        receiver: Box::new(Expr::Ident(TEMP.to_string(), span)),
        field: field.to_string(),
        span,
    };
    let assign = Expr::Assign {
        name: var.to_string(),
        value: Box::new(read),
        span,
    };
    eval_expr(&assign, env, trace).map(|_| ())
}
