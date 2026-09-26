//! Errors raised inside `grad` that name what went wrong -- the unsupported
//! expression form, a record where an array was expected, a parameter the loss
//! never reached -- instead of a generic "not supported" or a misleading
//! "the loss does not depend on the param" (demo-decision-model Q5).

use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::EvalError;

/// The error for an expression form `eval_tensor_expr` cannot put on the tape.
pub(crate) fn unsupported_form(expr: &Expr) -> EvalError {
    let form = match expr {
        Expr::RecordLit { .. } => "a record literal",
        Expr::StrLit(..) => "a string literal",
        Expr::If { .. } => "an `if` expression",
        Expr::While { .. } => "a `while` loop",
        Expr::For { .. } => "a `for` loop",
        Expr::Assign { .. } => "an assignment",
        _ => "this expression form",
    };
    EvalError::Unsupported(format!("grad: {form} is not supported inside grad()"))
}

/// The error for an identifier with no array binding in the traced scope. A
/// record is named as such: records are data, not differentiable values.
pub(crate) fn unbound_ident(name: &str, env: &Environment) -> EvalError {
    if env.get_record(name).is_some() {
        return EvalError::Unsupported(format!(
            "grad: '{name}' is a record; record values are not supported inside \
             grad() -- pass its fields as plain arrays"
        ));
    }
    EvalError::UndefinedVariable(name.to_string())
}

/// Optimizer steps: error if a requested, non-frozen parameter got no
/// gradient. Zero-filling it would make the step silently do nothing for that
/// weight -- the symptom a wrongly folded subexpression produces -- and REPL
/// users never see a notice, so this is a hard error, matching `grad()`.
pub(crate) fn require_reached(
    names: &[String],
    env: &Environment,
    reached: impl Fn(&str) -> bool,
) -> Result<(), EvalError> {
    let missing = names
        .iter()
        .find(|n| env.is_param(n) && !env.is_frozen(n) && !reached(n));
    match missing {
        None => Ok(()),
        Some(n) => Err(EvalError::Unsupported(format!(
            "optimizer step: '{n}' got no gradient -- the loss does not depend \
             on it; drop it from the parameter list or freeze() its model"
        ))),
    }
}
