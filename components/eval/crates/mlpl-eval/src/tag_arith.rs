//! Tag combination for arithmetic (`a + b`, etc.): the domain rules
//! split out of `tag_propagate` (per docs/code_metrics.md, split by
//! responsibility). Same-family operands keep the family, tagged +
//! untagged lets the tagged side win, and domain-mixing operands are
//! a `TypeMismatch` with a conversion hint.

use mlpl_core::{LossKind, ValueTag};
use mlpl_parser::{BinOpKind, Expr};

use crate::env::Environment;
use crate::tag_propagate::infer;
use mlpl_eval_types::EvalError;

const HINT_DOMAIN_MISMATCH: &str = "\
operands live in different typed-value domains. fix: convert one \
side first -- softmax(logits, axis) bridges Logit -> Probability, \
log lifts Probability -> LogProbability, and cross_entropy / mse / \
kl_divergence bridge predictions to Loss.";

pub(crate) fn arith(
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
    // Two losses combine into one whose specific kind is the lhs's;
    // differing kinds (e.g. CrossEntropy + Mse) keep the lhs's kind
    // by convention.
    lhs_kind
}

fn domain_mismatch_error(left: &str, right: &str) -> EvalError {
    EvalError::TypeMismatch {
        op: "binop".into(),
        expected: "compatible domain".into(),
        actual: format!("{left} + {right}"),
        hint: HINT_DOMAIN_MISMATCH.into(),
    }
}
