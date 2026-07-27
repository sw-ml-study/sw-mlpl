//! Program-level evaluation entry points: lex+parse already
//! happened; these run a `Vec<Expr>` to a final value.
//!
//! Extracted from `eval.rs` so the per-expression dispatch
//! (`eval_expr` + its big helpers) stays a separate concern
//! and the eval module fits under the sw-checklist
//! function-count budget.
//!
//! The four `eval_program*` functions all delegate to
//! `run_program`, which is the canonical loop. They differ
//! only in whether they accept a `Trace` and whether they
//! coerce the final `Value` to `DenseArray` (back-compat for
//! callers that pre-date the `Value` enum).

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

/// Evaluate a program (list of statements). Returns the last result as an array.
///
/// If the final value is a string, returns `EvalError::ExpectedArray`.
/// Use `eval_program_value` to handle both arrays and strings.
pub fn eval_program(stmts: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    eval_program_value(stmts, env)?.into_array()
}

/// Evaluate a program and return the final value (array or string).
pub fn eval_program_value(stmts: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    run_program(stmts, env, None)
}

/// Evaluate a program with tracing enabled. Returns the final array.
pub fn eval_program_traced(
    stmts: &[Expr],
    env: &mut Environment,
    trace: &mut Trace,
) -> Result<DenseArray, EvalError> {
    run_program(stmts, env, Some(trace))?.into_array()
}

/// Evaluate a program with tracing enabled and return the final
/// `Value`. Saga 31 step 006 -- the `-f` mode `Err(...) -> exit 1`
/// contract needs to inspect the value, so callers that also
/// want a trace use this variant instead of `eval_program_traced`.
pub fn eval_program_value_traced(
    stmts: &[Expr],
    env: &mut Environment,
    trace: &mut Trace,
) -> Result<Value, EvalError> {
    run_program(stmts, env, Some(trace))
}

fn run_program(
    stmts: &[Expr],
    env: &mut Environment,
    mut trace: Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if stmts.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    let mut result = None;
    for stmt in stmts {
        result = Some(eval_expr(stmt, env, &mut trace).map_err(map_top_level_signal)?);
    }
    result.ok_or(EvalError::EmptyInput)
}

/// Escaped control signals reaching the program top level become
/// user-facing errors: break/continue name their misuse, and a
/// stray `?` on an Err (no enclosing `u:` function) is loud
/// `unwrap` per docs/option-result-design.md.
fn map_top_level_signal(e: EvalError) -> EvalError {
    match e {
        EvalError::BreakSignal(_) => EvalError::LoopControlOutsideLoop { kind: "break" },
        EvalError::ContinueSignal => EvalError::LoopControlOutsideLoop { kind: "continue" },
        EvalError::ReturnSignal(v) => match *v {
            Value::Result { ok: false, payload } => EvalError::UnwrapOnErr {
                message: format!("{payload}"),
            },
            other => EvalError::ReturnSignal(Box::new(other)),
        },
        other => other,
    }
}
