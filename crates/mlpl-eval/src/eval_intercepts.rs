//! FnCall intercepts for the scripting-cluster + Result builtins.
//!
//! `eval_expr` would otherwise be a single fn over 800 lines.
//! Per docs/code_metrics.md (lib.rs / mod.rs are facades; behavior
//! in named files; split by responsibility), the early-return
//! `if let Expr::FnCall { name, .. }` chain for the scripting
//! and Result-typed builtins lives here. Each helper handles one
//! family; `try_intercept` is the dispatcher.
//!
//! Covered intercepts:
//! - `list_len(xs)` (saga 29 step 002)
//! - `ok(v)` / `err(v)` Result constructors (saga 29 step 012)
//! - `is_ok` / `is_err` / `unwrap` / `err_message` / `unwrap_or`
//!   Result accessors (saga 29 step 012)
//! - `print(v)` / `eprint(v)` (saga 31 step 001)
//! - `to_number(s)` / `to_int(s)` / `env(name)` (saga 31 step 002)
//! - `args()` / `list_get(xs, i)` (saga 31 step 003)

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;
use crate::value::{Value, value_kind};

/// Try matching `expr` against one of the FnCall intercepts in
/// this module. Returns `Some(result)` when matched, `None` if
/// the caller should continue with the main eval flow.
pub(crate) fn try_intercept(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    let Expr::FnCall { name, args, .. } = expr else {
        return None;
    };
    match name.as_str() {
        "list_len" => Some(eval_list_len(args, env, trace)),
        "ok" | "err" => Some(eval_result_ctor(name, args, env, trace)),
        "is_ok" | "is_err" | "unwrap" | "err_message" | "unwrap_or" => Some(
            crate::result_ops::eval_result_accessor(name, args, env, trace),
        ),
        "print" | "eprint" => Some(eval_print(name, args, env, trace)),
        "to_number" | "to_int" | "env" => Some(crate::result_ops::eval_string_to_result(
            name, args, env, trace,
        )),
        "args" => Some(eval_args(args, env)),
        "list_get" => Some(eval_list_get(args, env, trace)),
        _ => None,
    }
}

fn eval_list_len(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "list_len".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let v = eval_expr(&args[0], env, trace)?;
    match v {
        Value::StrList { items } => Ok(Value::Array(mlpl_array::DenseArray::from_scalar(
            items.len() as f64,
        ))),
        other => Err(EvalError::Unsupported(format!(
            "list_len: expected a string-list, got {}",
            value_kind(&other)
        ))),
    }
}

fn eval_result_ctor(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let payload = eval_expr(&args[0], env, trace)?;
    Ok(Value::Result {
        ok: name == "ok",
        payload: Box::new(payload),
    })
}

fn eval_print(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let v = eval_expr(&args[0], env, trace)?;
    if name == "print" {
        println!("{v}");
    } else {
        eprintln!("{v}");
    }
    Ok(v)
}

fn eval_args(args: &[Expr], env: &Environment) -> Result<Value, EvalError> {
    if !args.is_empty() {
        return Err(EvalError::BadArity {
            func: "args".into(),
            expected: 0,
            got: args.len(),
        });
    }
    Ok(Value::StrList {
        items: env.cli_args.clone(),
    })
}

fn eval_list_get(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    check_arity("list_get", args.len(), 2)?;
    let xs = eval_expr(&args[0], env, trace)?;
    let Value::StrList { items } = xs else {
        let msg = format!("list_get: expected a string-list, got {}", value_kind(&xs));
        return Err(EvalError::Unsupported(msg));
    };
    let i = parse_strlist_index(&args[1], env, trace)?;
    let (ok, s) = match items.get(i) {
        Some(s) => (true, s.clone()),
        None => (
            false,
            format!(
                "list_get: index {i} out of bounds (list has {} items)",
                items.len()
            ),
        ),
    };
    Ok(Value::Result {
        ok,
        payload: Box::new(Value::Str(s)),
    })
}

fn check_arity(name: &str, got: usize, expected: usize) -> Result<(), EvalError> {
    if got == expected {
        Ok(())
    } else {
        Err(EvalError::BadArity {
            func: name.into(),
            expected,
            got,
        })
    }
}

fn parse_strlist_index(
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<usize, EvalError> {
    let idx = eval_expr(arg, env, trace)?.into_array()?;
    if idx.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "list_get: index must be a scalar, got rank {}",
            idx.rank()
        )));
    }
    let v = idx.data()[0];
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "list_get: index must be a non-negative integer, got {v}"
        )));
    }
    Ok(v as usize)
}
