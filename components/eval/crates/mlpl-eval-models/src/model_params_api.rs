//! Reading and writing a built-in model's weights by ROLE:
//! `params(model)` (the parameter names, in order),
//! `get_param(model, role[, k])` and `set_param(model, role, value[, k])`.
//! A role is a layer's documented weight name (`W`, `b`, `Wq` .. `Wo`,
//! `table`, ...; see `model_roles`); `k` picks the k-th (0-based,
//! application order) layer owning that role in a chain. This is how
//! pretrained weights load into the built-in layers.

use crate::env_api::{EnvParams, EnvVars};
use crate::model_roles::param_for_role;
use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use mlpl_eval_env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Dispatch `params` / `get_param` / `set_param`; `None` for any other name.
pub fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
) -> Option<Result<Value, EvalError>> {
    Some(match name {
        "params" => eval_params(args, env),
        "get_param" => role_param(name, args, 2, env).and_then(|p| {
            let v = env.get(&p).cloned();
            v.map(Value::Array).ok_or(EvalError::UndefinedVariable(p))
        }),
        "set_param" => eval_set_param(args, env),
        _ => return None,
    })
}

/// `params(model)` -> the model's parameter names as a string list.
fn eval_params(args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    let [m] = args else {
        return Err(EvalError::BadArity {
            func: "params".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let items = crate::model_io::model_arg(m, env, "params")?.params();
    Ok(Value::StrList { items })
}

/// Resolve `model, role, ..fixed.., [k]` to the parameter name. `fixed` is
/// the arity without the optional trailing `k`; the role is `args[1]`.
fn role_param(
    func: &str,
    args: &[Expr],
    fixed: usize,
    env: &mut Environment,
) -> Result<String, EvalError> {
    if !(fixed..=fixed + 1).contains(&args.len()) {
        let got = args.len();
        return Err(EvalError::BadArity {
            func: func.into(),
            expected: fixed,
            got,
        });
    }
    let spec = crate::model_io::model_arg(&args[0], env, func)?;
    let Value::Str(role) = mlpl_eval_env::dispatch_hook::eval_or_err(&args[1], env, &mut None)?
    else {
        return Err(EvalError::Unsupported(format!(
            "{func}: the role must be a string such as \"W\""
        )));
    };
    let k = args
        .get(fixed)
        .map(|e| crate::model_dispatch_scalar::scalar_usize(e, env, func));
    let k = k.transpose()?.unwrap_or(0);
    param_for_role(&spec, &role, k).map_err(|m| EvalError::Unsupported(format!("{func}: {m}")))
}

/// `set_param(model, role, value[, k])`: shape-checked write that, like an
/// optimizer update, persists past the enclosing user-function frame.
/// Returns scalar 0.
fn eval_set_param(args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    let name = role_param("set_param", args, 3, env)?;
    let value =
        mlpl_eval_env::dispatch_hook::eval_or_err(&args[2], env, &mut None)?.into_array()?;
    let want = env
        .get(&name)
        .map(|a| a.shape().dims().to_vec())
        .unwrap_or_default();
    if value.shape().dims() != want.as_slice() {
        let got = value.shape().dims().to_vec();
        return Err(EvalError::Unsupported(format!(
            "set_param: '{name}' has shape {want:?}; got {got:?}"
        )));
    }
    if env.call_depth > 0 {
        env.global_writes
            .push((name.clone(), Value::Array(value.clone())));
    }
    env.set_param(name, value);
    Ok(Value::Array(DenseArray::from_scalar(0.0)))
}
