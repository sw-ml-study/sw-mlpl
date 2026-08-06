//! Top-level FnCall dispatch for `eval::eval_expr`.
//!
//! Saga 33 step 023 split the FnCall name-driven block out of
//! `eval::eval_expr` to retire that file's File-LOC FAIL.
//! `try_dispatch` walks the family modules in order
//! (`fncall_models`, `fncall_axes`, `fncall_arrays`) and then
//! falls through to the loader / tools cluster handled inline
//! here. Each family returns `Option<Result<Value, EvalError>>`
//! so the caller chains them with `Option::or_else`.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub(crate) fn try_dispatch(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    let Expr::FnCall { name, args, span } = expr else {
        return None;
    };
    crate::fncall_models::try_dispatch(name, args, env, trace)
        .or_else(|| crate::fncall_axes::try_dispatch(name, args, env, trace))
        .or_else(|| crate::fncall_arrays::try_dispatch(name, args, env, trace, span))
        .or_else(|| crate::fncall_engram::try_dispatch(name, args, env, trace))
        .or_else(|| crate::fncall_gen::try_dispatch(name, args, env, trace))
        .or_else(|| crate::fncall_events::try_dispatch(name, args, env, trace))
        .or_else(|| crate::fncall_globals::try_dispatch(name, args, env, trace))
        .or_else(|| crate::fncall_values::try_dispatch(name, args, env, trace, span))
        .or_else(|| try_loaders(name, args, env, trace))
        .or_else(|| try_tools(name, args, env, trace, span))
}

fn try_loaders(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "load" | "load_preloaded" => Some(eval_load_call(name, args, env)),
        "fetch_dataset" => Some(eval_fetch_dataset(args, env)),
        "reduce" => Some(crate::eval_reduce::eval_reduce(args, env, trace)),
        _ => None,
    }
}

fn eval_load_call(name: &str, args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let Expr::StrLit(path, _) = &args[0] else {
        return Err(EvalError::Unsupported(format!(
            "{name}: argument must be a string literal"
        )));
    };
    if name == "load" {
        crate::loader::eval_load(env, path)
    } else {
        crate::loader::eval_load_preloaded(path)
    }
}

fn eval_fetch_dataset(args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "fetch_dataset".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let Expr::StrLit(dataset, _) = &args[0] else {
        return Err(EvalError::Unsupported(
            "fetch_dataset: argument must be a string literal".into(),
        ));
    };
    crate::loader::eval_fetch_dataset(env, dataset)
}

fn try_tools(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    if let Some(r) = crate::tokenizer::dispatch(name, args, env, trace) {
        return Some(r);
    }
    match name {
        "llm_call" => Some(crate::llm_dispatch::dispatch(args, env, trace)),
        "emit_frame" => Some(crate::fncall_trace::eval_emit_frame(args, env, trace)),
        "compare" => Some(crate::experiment_compare::dispatch_compare(args, env)),
        "map_ok" | "and_then" | "or_else" => Some(crate::fncall_combinators::eval_combinator(
            name, args, env, trace,
        )),
        "experiment_metric" => Some(crate::experiment_compare::eval_experiment_metric(args, env)),
        "momentum_sgd" | "adam" => {
            Some(crate::grad_optim::eval_optim(name, args, env, trace, span))
        }
        _ => {
            crate::eval_ops::eval_analysis_helper(name, args, env, trace).map(|r| r.map(Value::Str))
        }
    }
}
