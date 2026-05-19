//! Result<val, err> accessor dispatch (Saga 29 step 012).
//!
//! Holds the small helpers for `is_ok`, `is_err`, `unwrap`,
//! `err_message`, and `unwrap_or`. Lives in its own module so
//! eval.rs stays under the per-module function-count budget.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;
use crate::value::{Value, value_kind};

impl Environment {
    pub fn set_result(&mut self, name: String, ok: bool, payload: Value) {
        self.results.insert(name, (ok, payload));
    }
    #[must_use]
    pub fn get_result(&self, name: &str) -> Option<&(bool, Value)> {
        self.results.get(name)
    }
}

/// Dispatch one of the Result accessors. The arity check and
/// receiver-kind check live here so the eval.rs early-return
/// only has to recognize the function name.
pub(crate) fn eval_result_accessor(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let want = if name == "unwrap_or" { 2 } else { 1 };
    if args.len() != want {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: want,
            got: args.len(),
        });
    }
    let recv = eval_expr(&args[0], env, trace)?;
    let Value::Result { ok, payload } = recv else {
        return Err(EvalError::NotAResult {
            receiver_kind: value_kind(&recv),
            accessor: static_name(name),
        });
    };
    match name {
        "is_ok" => Ok(scalar_bool(ok)),
        "is_err" => Ok(scalar_bool(!ok)),
        "unwrap" if ok => Ok(*payload),
        "unwrap" => Err(EvalError::UnwrapOnErr {
            message: format!("{payload}"),
        }),
        "err_message" if !ok => Ok(*payload),
        "err_message" => Err(EvalError::Unsupported(format!(
            "err_message: receiver is Ok({payload}), not Err(_)"
        ))),
        "unwrap_or" if ok => Ok(*payload),
        "unwrap_or" => eval_expr(&args[1], env, trace),
        _ => unreachable!("dispatcher guard kept us in the accessor set"),
    }
}

fn scalar_bool(b: bool) -> Value {
    Value::Array(DenseArray::from_scalar(if b { 1.0 } else { 0.0 }))
}

fn static_name(name: &str) -> &'static str {
    match name {
        "is_ok" => "is_ok",
        "is_err" => "is_err",
        "unwrap" => "unwrap",
        "err_message" => "err_message",
        "unwrap_or" => "unwrap_or",
        _ => "result-accessor",
    }
}
