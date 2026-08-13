//! Result<val, err> accessor dispatch (Saga 29 step 012).
//!
//! Holds the small helpers for `is_ok`, `is_err`, `unwrap`,
//! `err_message`, and `unwrap_or`. Lives in its own module so
//! eval.rs stays under the per-module function-count budget.

use crate::env_api::*;
use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::{Value, value_kind};

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
        "get_value" => project_option(*payload, ok, "get_value"),
        "get_error" => project_option(*payload, !ok, "get_error"),
        // `expr?` desugars to check(expr): ok unwraps; err
        // early-returns the WHOLE Result via the return signal
        // (call_user_fn catches it; a stray top-level signal is
        // mapped to loud UnwrapOnErr by run_program).
        "check" if ok => Ok(*payload),
        "check" => Err(EvalError::ReturnSignal(Box::new(Value::Result {
            ok: false,
            payload,
        }))),
        _ => unreachable!("dispatcher guard kept us in the accessor set"),
    }
}

/// APL2 zilde-flavored Option projection (Game of Life saga era,
/// docs/option-result-design.md): the wanted side of a Result as a
/// 0-or-1 element vector -- `[]` when absent, `[payload]` when
/// present -- so `tally` is `is_some` and
/// `take(concat(get_value(r), [d]), 0, 0)` is `unwrap_or`. Scalar
/// payloads only until Stage 6 nested arrays bring `enclose`.
fn project_option(payload: Value, present: bool, accessor: &str) -> Result<Value, EvalError> {
    if !present {
        return Ok(Value::Array(DenseArray::new(Shape::new(vec![0]), vec![])?));
    }
    match payload {
        Value::Array(a) if a.rank() == 0 => Ok(Value::Array(DenseArray::new(
            Shape::new(vec![1]),
            vec![a.data()[0]],
        )?)),
        other => Err(EvalError::Unsupported(format!(
            "{accessor}: boxing a non-scalar payload ({}) as an Option needs Stage 6 enclose; use unwrap/err_message instead",
            mlpl_eval_types::value_kind(&other)
        ))),
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
        "get_value" => "get_value",
        "get_error" => "get_error",
        "check" => "check",
        _ => "result-accessor",
    }
}

/// `try { body } catch <binding> { handler }` (spike step 011).
/// Runs `body`; a HARD error binds `binding` to the canonical
/// `{kind, message}` record and yields the handler's value.
/// Control-flow signals (break/continue/return) and `err(...)`
/// VALUES are not errors here -- signals re-raise, values flow.
pub(crate) fn eval_try_catch(
    body: &[Expr],
    binding: &str,
    handler: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut last = Value::Array(DenseArray::from_scalar(0.0));
    for stmt in body {
        match eval_expr(stmt, env, trace) {
            Ok(v) => last = v,
            Err(
                sig @ (EvalError::BreakSignal(_)
                | EvalError::ContinueSignal
                | EvalError::ReturnSignal(_)
                | EvalError::ExitRequested(_)),
            ) => return Err(sig),
            Err(e) => return run_catch_handler(&e, binding, handler, env, trace),
        }
    }
    Ok(last)
}

/// Bind the error record and evaluate the handler body.
fn run_catch_handler(
    e: &EvalError,
    binding: &str,
    handler: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert(
        "kind".to_string(),
        Value::Str(mlpl_eval_types::error_kind(e).to_string()),
    );
    fields.insert("message".to_string(), Value::Str(format!("{e}")));
    env.set_record(binding.to_string(), fields);
    let mut last = Value::Array(DenseArray::from_scalar(0.0));
    for stmt in handler {
        last = eval_expr(stmt, env, trace)?;
    }
    Ok(last)
}
