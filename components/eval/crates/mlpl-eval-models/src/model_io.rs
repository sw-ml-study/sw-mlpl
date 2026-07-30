//! `save_model` / `load_model`: persist a trained model (its spec +
//! every parameter value) to a JSON snapshot and restore it. Lets a
//! fine-tuned model be reused -- train once, save, then load to
//! compare/serve -- instead of retraining every run.

use crate::env_api::{EnvModels, EnvParams, EnvVars};
use std::fs;

use mlpl_array::{DenseArray, Shape};
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_core::snapshot::{ModelSnapshot, ParamEntry};
use mlpl_parser::Expr;

use mlpl_eval_env::Environment;
use mlpl_eval_types::{EvalError, Value};

fn path_arg(expr: &Expr, func: &str) -> Result<String, EvalError> {
    match expr {
        Expr::StrLit(p, _) => Ok(p.clone()),
        _ => Err(EvalError::Unsupported(format!(
            "{func}: path must be a string literal"
        ))),
    }
}

fn model_arg(expr: &Expr, env: &mut Environment, func: &str) -> Result<ModelSpec, EvalError> {
    // A bare model identifier lives in `env.models` (not `vars`), so
    // resolve it by name first -- matching how the other model
    // builtins (apply, grad's collect_params) reach a model. Fall back
    // to evaluating an inline model expression.
    if let Expr::Ident(name, _) = expr
        && let Some(m) = env.get_model(name)
    {
        return Ok(m.clone());
    }
    match mlpl_eval_env::dispatch_hook::eval_or_err(expr, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(format!(
            "{func}: first argument must be a model"
        ))),
    }
}

/// `save_model(model, "path")` -- write the spec + every param value to
/// a JSON snapshot. Returns the model unchanged (pass-through).
pub fn eval_save_model(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "save_model".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let spec = model_arg(&args[0], env, "save_model")?;
    let path = path_arg(&args[1], "save_model")?;
    let mut params = Vec::new();
    for name in spec.params() {
        let arr = env
            .get(&name)
            .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))?;
        params.push(ParamEntry {
            dims: arr.shape().dims().to_vec(),
            data: arr.data().to_vec(),
            name,
        });
    }
    let snap = ModelSnapshot {
        version: ModelSnapshot::VERSION,
        spec: spec.clone(),
        params,
    };
    let json = serde_json::to_string(&snap)
        .map_err(|e| EvalError::Unsupported(format!("save_model: serialize: {e}")))?;
    fs::write(&path, json)
        .map_err(|e| EvalError::Unsupported(format!("save_model: write {path}: {e}")))?;
    Ok(spec)
}

/// `load_model("path")` -- restore a snapshot: set every param value in
/// the environment and return the spec. Assign it to bind the model:
/// `m = load_model("path")`.
pub fn eval_load_model(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "load_model".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let path = path_arg(&args[0], "load_model")?;
    let json = fs::read_to_string(&path)
        .map_err(|e| EvalError::Unsupported(format!("load_model: read {path}: {e}")))?;
    let snap: ModelSnapshot = serde_json::from_str(&json)
        .map_err(|e| EvalError::Unsupported(format!("load_model: parse {path}: {e}")))?;
    for p in snap.params {
        let arr = DenseArray::new(Shape::new(p.dims), p.data)
            .map_err(|e| EvalError::Unsupported(format!("load_model: array '{}': {e}", p.name)))?;
        env.set_param(p.name, arr);
    }
    Ok(snap.spec)
}
