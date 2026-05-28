//! Thin wrapper around `mlpl-models-llm::llm_call_inner`.
//! Evaluates each Expr arg to a String via `eval_expr`, then
//! delegates the runtime call to the sub-crate and re-wraps
//! the reply in `Value::Str` at the boundary.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use crate::eval::eval_expr;
use mlpl_eval_types::Value;

pub(crate) fn dispatch(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut strs: Vec<String> = Vec::with_capacity(args.len());
    for arg in args {
        match eval_expr(arg, env, trace)? {
            Value::Str(s) => strs.push(s),
            _ => return Err(EvalError::ExpectedString),
        }
    }
    let reply = mlpl_models_llm::llm_call_inner(&strs)?;
    Ok(Value::Str(reply))
}
