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

use mlpl_core::{LossKind, ValueTag};
use mlpl_parser::{BinOpKind, Expr};

use crate::env::Environment;
use crate::error::EvalError;

const HINT_DOMAIN_MISMATCH: &str = "\
operands live in different typed-value domains. fix: convert one \
side first -- softmax(logits, axis) bridges Logit -> Probability, \
log lifts Probability -> LogProbability, and cross_entropy / mse / \
kl_divergence bridge predictions to Loss.";

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

fn arith(
    op: &BinOpKind,
    lhs: &Expr,
    rhs: &Expr,
    env: &Environment,
) -> Result<Option<ValueTag>, EvalError> {
    let _ = op;
    let lt = infer(lhs, env);
    let rt = infer(rhs, env);
    match (lt, rt) {
        (None, None) => Ok(None),
        (Some(t), None) | (None, Some(t)) => Ok(Some(t)),
        (Some(a), Some(b)) => combine_pair(&a, &b),
    }
}

fn combine_pair(a: &ValueTag, b: &ValueTag) -> Result<Option<ValueTag>, EvalError> {
    match (a, b) {
        (ValueTag::Logit, ValueTag::Logit) => Ok(Some(ValueTag::Logit)),
        (ValueTag::Loss { kind: ka }, ValueTag::Loss { .. }) => Ok(Some(ValueTag::Loss {
            kind: loss_kind_join(*ka),
        })),
        // Same singleton tag on both sides: passthrough.
        (x, y) if x == y => Ok(Some(x.clone())),
        _ => Err(domain_mismatch_error(a.display_name(), b.display_name())),
    }
}

fn loss_kind_join(lhs_kind: LossKind) -> LossKind {
    // Two losses combine into one whose specific kind is the
    // lhs's; differing kinds (e.g. CrossEntropy + Mse) keep the
    // lhs's kind by convention. A future step may upgrade this
    // to LossKind::Custom for mixed kinds.
    lhs_kind
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

fn domain_mismatch_error(left: &str, right: &str) -> EvalError {
    EvalError::TypeMismatch {
        op: "binop".into(),
        expected: "compatible domain".into(),
        actual: format!("{left} + {right}"),
        hint: HINT_DOMAIN_MISMATCH.into(),
    }
}
