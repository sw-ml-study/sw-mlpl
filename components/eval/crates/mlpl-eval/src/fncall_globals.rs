//! `global_set(name, value)` -- the EXPLICIT global-state
//! escape hatch. Binding hygiene stays the default (frames
//! restore); a spelled-out global write binds immediately AND
//! is recorded outside the frame snapshot, so the frame helper
//! replays it after each restore -- surviving to the top level
//! without ever letting implicit assignment leak.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "global_set" => Some(eval_global_set(args, env, trace)),
        _ => None,
    }
}

fn eval_global_set(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [name_arg, value_arg] = args else {
        return Err(EvalError::BadArity {
            func: "global_set".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let name = match crate::eval::eval_expr(name_arg, env, trace)? {
        Value::Str(s) => s,
        other => {
            return Err(EvalError::Unsupported(format!(
                "global_set: the first argument is the name of the binding (a string) -- got {}",
                mlpl_eval_types::value_kind(&other)
            )));
        }
    };
    if !crate::expunge::well_formed(&name) {
        return Err(EvalError::Unsupported(format!(
            "global_set: `{name}` is not a bindable name"
        )));
    }
    let value = crate::eval::eval_expr(value_arg, env, trace)?;
    bind_value(env, &name, value.clone());
    if env.call_depth > 0 {
        env.global_writes.push((name, value.clone()));
    }
    Ok(value)
}

/// Kind-routed binding, mirroring assignment: clear the name
/// from every table, then set it in the right one.
pub(crate) fn bind_value(env: &mut Environment, name: &str, value: Value) {
    env.clear_binding(name);
    match value {
        Value::Array(a) => env.set(name.to_string(), a),
        Value::Str(s) => env.set_string(name.to_string(), s),
        Value::Record { fields } => env.set_record(name.to_string(), fields),
        Value::StrList { items } => env.set_string_list(name.to_string(), items),
        Value::Result { ok, payload } => env.set_result(name.to_string(), ok, *payload),
        Value::BuiltinRef { name: target } | Value::UserFnRef { name: target } => {
            env.set_builtin_ref(name.to_string(), target);
        }
        Value::Model(m) => {
            env.models.insert(name.to_string(), m);
        }
        Value::Tokenizer(t) => env.set_tokenizer(name.to_string(), t),
        Value::GenState(g) => {
            env.gen_states.insert(name.to_string(), *g);
        }
        v @ Value::DeviceTensor { .. } => env.set_device_tensor(name.to_string(), v),
    }
}
