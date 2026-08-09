//! Native-extension dispatch. A colon-qualified `FnCall` whose
//! name is registered (`namespace:function`) is invoked through
//! the extension registry, marshaling the V1 scalar set across
//! the boundary (see `ext_marshal`) with panics CONTAINED. A
//! recoverable domain failure becomes an `err(...)` Result; a
//! contained panic or a boundary-contract violation (bad arity /
//! unsupported argument) is a hard `EvalError::ExtensionError`.

use mlpl_extension_abi::{ExtFnDesc, call_contained};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

/// Dispatch a colon-qualified call to a registered extension.
/// Returns `None` if the name is not a registered extension
/// function (so the caller falls through to user-fn routing).
pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    let desc = mlpl_extension_registry::lookup(name)?;
    Some(invoke(name, &desc, args, env, trace))
}

fn invoke(
    name: &str,
    desc: &ExtFnDesc,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != desc.arity {
        return Err(EvalError::ExtensionError {
            function: name.to_string(),
            message: format!("expected {} argument(s), got {}", desc.arity, args.len()),
        });
    }
    let mut ext_args = Vec::with_capacity(args.len());
    for a in args {
        ext_args.push(crate::ext_marshal::to_ext(name, eval_expr(a, env, trace)?)?);
    }
    crate::ext_marshal::finish(name, call_contained(desc.func, &ext_args))
}
