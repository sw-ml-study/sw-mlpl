//! `record_keys(record)` -> string-list of the record's keys in
//! deterministic (sorted) order -- records are BTreeMap-backed,
//! so `keys()` is already sorted. A non-record argument is a hard
//! error (a type error, like the other record accessors). Lets a
//! program discover a parsed record's field names (e.g. the tensor
//! names in a safetensors JSON header).

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    if name != "record_keys" {
        return None;
    }
    Some(record_keys(args, env, trace))
}

fn record_keys(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "record_keys".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let Value::Record { fields } = crate::eval::eval_expr(arg, env, trace)? else {
        return Err(EvalError::Unsupported(
            "record_keys: the argument must be a record".into(),
        ));
    };
    Ok(Value::StrList {
        items: fields.keys().cloned().collect(),
    })
}
