//! Records inside `grad` (demo-decision-model Q5). Records are DATA: a
//! parameter is identified by name, never by a record field, so a field
//! read is a constant leaf and the gradient flows through the surrounding
//! ops (a record OF weights is therefore not trainable -- pass the params
//! themselves). A record-valued `u:` argument binds as a real record for
//! the body's field reads, inside an undo-log frame so it cannot leak.

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::{EvalError, Value};

/// `record.field` inside `grad`: evaluated eagerly over the traced-scope
/// overlay and inserted as an untracked constant leaf.
pub(crate) fn field_leaf(
    expr: &Expr,
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let v = crate::grad_const::eval_const_arg(expr, env, params)?;
    Ok(Tensor::leaf(Rc::clone(tape), v, false))
}

/// Bind every record-valued argument (a record variable or a record
/// literal) to its parameter name as a real record, removing the name from
/// the traced scope `local` so no tensor shadows it. Returns the names
/// bound; when non-empty, an undo-log frame is open and the caller must
/// close it with `frame_exit` after tracing the body.
pub(crate) fn bind_record_args(
    params: &[String],
    args: &[Expr],
    env: &mut Environment,
    local: &mut HashMap<String, Tensor>,
) -> Result<Vec<String>, EvalError> {
    // Evaluate first, bind second: an error leaves no frame open.
    let mut records = Vec::new();
    for (p, arg) in params.iter().zip(args) {
        if !is_record_arg(arg, env, local) {
            continue;
        }
        if let Value::Record { fields } = crate::eval::eval_expr(arg, env, &mut None)? {
            records.push((p.clone(), fields));
        }
    }
    Ok(bind_in_frame(records, env, local))
}

/// Open a frame (when there is anything to bind) and bind each record under
/// its parameter name, out of the traced scope. Returns the names bound.
fn bind_in_frame(
    records: Vec<(String, BTreeMap<String, Value>)>,
    env: &mut Environment,
    local: &mut HashMap<String, Tensor>,
) -> Vec<String> {
    if !records.is_empty() {
        env.frame_journal.push(Default::default());
    }
    let mut bound = Vec::with_capacity(records.len());
    for (p, fields) in records {
        env.clear_binding(&p);
        env.set_record(p.clone(), fields);
        local.remove(&p);
        bound.push(p);
    }
    bound
}

/// A record variable (not shadowed by a traced tensor) or a record literal.
fn is_record_arg(arg: &Expr, env: &Environment, local: &HashMap<String, Tensor>) -> bool {
    match arg {
        Expr::Ident(n, _) => !local.contains_key(n) && env.get_record(n).is_some(),
        Expr::RecordLit { .. } => true,
        _ => false,
    }
}
