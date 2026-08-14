//! Top-level FnCall dispatch for `eval::eval_expr`.
//!
//! `try_dispatch` forwards a call down the `DISPATCHERS` registry --
//! an ordered list of builtin-family dispatchers -- with `find_map`,
//! stopping at the first family that handles the name (returns
//! `Some`). Every family conforms to one uniform `Dispatcher`
//! signature `(name, args, env, trace, span)`, so the registry is
//! data: adding a family is one row, not a link in a hand-written
//! `or_else` chain. Config (the list) / dispatch (the loop) / handlers
//! (the family fns) are separated. First-match order is preserved.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

/// A builtin-family dispatcher: given a call's `name`/`args` (plus the
/// eval context and the call `span`), returns `Some(result)` if this
/// family handles the name, else `None`. Every family conforms to this
/// one signature -- a uniform handler contract -- so the registry can
/// hold them all; families that do not need `span` ignore it.
type Dispatcher = fn(
    &str,
    &[Expr],
    &mut Environment,
    &mut Option<&mut Trace>,
    &mlpl_core::Span,
) -> Option<Result<Value, EvalError>>;

/// The dispatch REGISTRY: the ordered list of builtin families. The
/// call is forwarded down the list until a family handles it (returns
/// `Some`) -- first match wins. Adding a family is one row here, not a
/// new link in a hand-written `or_else` chain.
const DISPATCHERS: &[Dispatcher] = &[
    crate::fncall_models::try_dispatch,
    crate::fncall_bytes::try_dispatch,
    crate::fncall_axes::try_dispatch,
    crate::fncall_arrays::try_dispatch,
    crate::fncall_engram::try_dispatch,
    crate::fncall_gen::try_dispatch,
    crate::fncall_gen_controls::try_dispatch,
    crate::fncall_record::try_dispatch,
    crate::fncall_record_keys::try_dispatch,
    crate::fncall_events::try_dispatch,
    crate::fncall_globals::try_dispatch,
    crate::fncall_fs::try_dispatch,
    crate::fncall_run::try_dispatch,
    crate::fncall_hof::try_dispatch,
    crate::fncall_json::try_dispatch,
    crate::fncall_toml::try_dispatch,
    crate::fncall_native::try_dispatch,
    crate::fncall_string::try_dispatch,
    crate::fncall_string_ops::try_dispatch,
    crate::fncall_values::try_dispatch,
    try_loaders,
    try_tools,
];

pub(crate) fn try_dispatch(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    let Expr::FnCall { name, args, span } = expr else {
        return None;
    };
    DISPATCHERS
        .iter()
        .find_map(|dispatch| dispatch(name, args, env, trace, span))
}

fn try_loaders(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
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
