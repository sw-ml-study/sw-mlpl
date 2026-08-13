//! Saga 23 step 005: tag propagation through arithmetic,
//! transpose, reshape, reductions, negation, and identifier
//! aliases.
//!
//! Producer rules from steps 002-003 take precedence:
//! `auto_tag::for_assign` runs first, and propagation only fires
//! when no producer matches. The propagation table:
//!
//! - Same-family arithmetic (Logit + Logit, Loss + Loss):
//!   keep the family.
//! - Tagged + Untagged: the tagged side wins.
//! - Untagged + Untagged: untagged.
//! - Domain-mixing arithmetic (Logit + Probability,
//!   Loss + Probability, etc.): TypeMismatch with hint.
//! - transpose / reshape_labeled: preserve.
//! - reshape: clear (shape reflow loses semantic identity).
//! - mean / reduce_add / reduce_mul / argmax: Loss survives,
//!   everything else clears.
//! - Unary negation: preserve.
//! - Bare identifier: copy from the side table.

use crate::env_api::*;
use mlpl_core::ValueTag;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::tag_arith::arith;
use mlpl_eval_types::EvalError;

/// Run propagation on the right-hand side of an assignment when
/// no producer rule from `auto_tag::for_assign` matched. Returns
/// `Ok(Some(tag))` when the rhs propagates a tag, `Ok(None)` to
/// leave the binding untagged, or `Err` for domain mismatches.
pub(crate) fn propagate(value: &Expr, env: &Environment) -> Result<Option<ValueTag>, EvalError> {
    match value {
        Expr::Ident(name, _) => Ok(env.get_tag(name).cloned()),
        Expr::UnaryNeg { operand, .. } => Ok(infer(operand, env)),
        Expr::BinOp { op, lhs, rhs, .. } => arith(op, lhs, rhs, env),
        Expr::FnCall { name, args, .. } => fncall_propagate(name, args, env),
        _ => Ok(None),
    }
}

/// Best-effort tag for a sub-expression with no error surfacing.
/// Used by predicates and by recursive propagation walks: a
/// domain-mismatch deeper in the tree returns None here, and the
/// surrounding op decides whether to propagate or error.
pub(crate) fn infer(value: &Expr, env: &Environment) -> Option<ValueTag> {
    match value {
        Expr::Ident(name, _) => env.get_tag(name).cloned(),
        Expr::FnCall { .. } => crate::auto_tag::for_assign(value, env),
        Expr::UnaryNeg { operand, .. } => infer(operand, env),
        Expr::BinOp { op, lhs, rhs, .. } => arith(op, lhs, rhs, env).ok().flatten(),
        _ => None,
    }
}

fn fncall_propagate(
    name: &str,
    args: &[Expr],
    env: &Environment,
) -> Result<Option<ValueTag>, EvalError> {
    let Some(first) = args.first() else {
        return Ok(None);
    };
    let in_tag = infer(first, env);
    Ok(match name {
        "transpose" | "reshape_labeled" | "label" | "relabel" => in_tag,
        "reshape" => None,
        "mean" | "reduce_add" | "reduce_mul" | "argmax" => reduce_keep(in_tag),
        _ => None,
    })
}

fn reduce_keep(in_tag: Option<ValueTag>) -> Option<ValueTag> {
    match in_tag {
        Some(ValueTag::Loss { kind }) => Some(ValueTag::Loss { kind }),
        _ => None,
    }
}
