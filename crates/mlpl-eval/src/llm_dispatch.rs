//! Saga 19 step 001: `llm_call(url, prompt, model)`
//! eval-side dispatcher.
//!
//! Evaluates the three string arguments via
//! `eval_expr`, calls `mlpl_runtime::call_ollama` for
//! the actual HTTP exchange, and wraps the reply in
//! `Value::Str`. The runtime helper owns URL
//! normalization, JSON parsing, and timeout handling;
//! this shim is just the `Expr -> String` adapter
//! plus the `RuntimeError -> EvalError` lift.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;
use crate::value::Value;

pub(crate) fn dispatch(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "llm_call".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let mut strs: [String; 3] = [String::new(), String::new(), String::new()];
    for (i, slot) in strs.iter_mut().enumerate() {
        match eval_expr(&args[i], env, trace)? {
            Value::Str(s) => *slot = s,
            _ => return Err(EvalError::ExpectedString),
        }
    }
    let [url, prompt, model] = strs;
    let reply = mlpl_runtime::call_ollama(&url, &prompt, &model)
        .map_err(|e| EvalError::Unsupported(format!("{e}")))?;
    Ok(Value::Str(reply))
}
