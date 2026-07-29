//! Saga 23 step 004: predicate-checked consumers + tutoring hints.
//!
//! Educational-first ranking shapes this module. Type errors are
//! not just "tag mismatch" -- they are tutor messages that name
//! the most likely cause and a concrete fix, so the student
//! treats them as a learning moment rather than an obstacle.
//!
//! Untagged arguments always pass (`gradual-typing additivity`):
//! a user who hasn't yet adopted typed values keeps the existing
//! shape-only behavior with no surprises.

use crate::env_api::*;
use mlpl_core::ValueTag;
use mlpl_parser::Expr;

use crate::env::Environment;
use mlpl_eval_types::EvalError;

// Hint catalog. Each hint follows a consistent four-part shape
// for educational clarity:
//   1) what the op expected and why,
//   2) what it got (named in actual + repeated in prose),
//   3) likely cause,
//   4) one or two concrete fixes with copy-pasteable MLPL.
//
// Copy-pasteable MLPL is canonicalized: softmax takes an axis
// (`softmax(x, 1)`), cross_entropy is `cross_entropy(logits, y)`.
// Any future-only fix references the saga that will ship it
// (e.g. Saga 24 for nll). The tutoring_hints contract doc
// (contracts/eval-contract/tutoring-hints.md) catalogs all
// hints in one place and locks the canonical fix syntax.

const HINT_LOGIT_GOT_PROBABILITY: &str = "\
this consumer expects Logit (raw scores) because cross_entropy / \
sample / top_k assume an unnormalized score per class. \
you got Probability. likely cause: you applied softmax already. \
fix: drop the softmax and pass the original Logit -- \
`loss = cross_entropy(logits, y)` -- or, once Saga 24 ships nll, \
`loss = nll(probs, y)`. double-softmax inside cross_entropy is \
the canonical NaN factory at scale.";

const HINT_LOGIT_GOT_LOG_PROBABILITY: &str = "\
this consumer expects Logit (raw scores) but the argument is \
already log-softmaxed. likely cause: you applied log_softmax \
upstream, or composed `log(softmax(x, 1))`. \
fix: drop the log_softmax and pass the original Logit -- \
`loss = cross_entropy(logits, y)` -- or, on log-probs, use \
`loss = nll(log_probs, y)` once Saga 24 ships nll. \
recovering Probability via `exp(log_probs)` is also valid for \
viz / sampling.";

const HINT_LOGIT_GOT_LABELS: &str = "\
cross_entropy expects Logit as the first arg and Labels as the \
second; you swapped them. likely cause: arg order confusion. \
fix: swap the arguments -- `loss = cross_entropy(logits, y)` -- \
where `logits` is `[N, V]` (or `[B, T, V]`) and `y` is `[N]` \
integer-valued class indices.";

const HINT_LOSS_GOT_PROBABILITY: &str = "\
adam / momentum_sgd expect a scalar Loss as the first arg \
(they minimize it). you got Probability. likely cause: you \
forgot the loss function. fix: wrap predictions in a loss term \
first -- `loss = cross_entropy(logits, y)`; adam(loss, params, \
lr, b1, b2, eps)`. mse / kl_divergence will work too once they \
ship as builtins.";

const HINT_LOSS_GOT_LOGIT: &str = "\
adam / momentum_sgd expect a scalar Loss but you passed a \
Logit (predictions, not the loss). fix: produce the loss term \
first, then pass it -- `loss = cross_entropy(logits, y)` then \
`adam(loss, params, lr, b1, b2, eps)`.";

const HINT_LOSS_GOT_OTHER: &str = "\
adam / momentum_sgd expect a scalar Loss-tagged value. likely \
cause: the first arg came from a non-loss op. fix: produce a \
Loss with cross_entropy / mse / kl_divergence, e.g. \
`loss = cross_entropy(logits, y)`, and pass `loss` as the \
first arg.";

/// Look up the tag carried by an expression in the current
/// environment. Named bindings consult the side table; FnCall
/// expressions delegate to the auto-tag dispatcher (so an inline
/// `softmax(x, axis)` resolves to Probability without needing a
/// binding).
pub(crate) fn arg_tag(arg: &Expr, env: &Environment) -> Option<ValueTag> {
    match arg {
        Expr::Ident(name, _) => env.get_tag(name).cloned(),
        Expr::FnCall { .. } => crate::auto_tag::for_assign(arg, env),
        _ => None,
    }
}

/// Predicate for ops that consume a Logit (cross_entropy,
/// sample, top_k). Returns Ok if the arg is tagged Logit or has
/// no tag at all; raises TypeMismatch with a tutoring hint
/// otherwise.
pub(crate) fn check_logit_consumer(
    op: &str,
    arg: &Expr,
    env: &Environment,
) -> Result<(), EvalError> {
    match arg_tag(arg, env) {
        None | Some(ValueTag::Logit) => Ok(()),
        Some(tag) => {
            let hint = match &tag {
                ValueTag::Probability => HINT_LOGIT_GOT_PROBABILITY,
                ValueTag::LogProbability => HINT_LOGIT_GOT_LOG_PROBABILITY,
                ValueTag::Labels { .. } if op == "cross_entropy" => HINT_LOGIT_GOT_LABELS,
                _ => return Ok(()),
            };
            Err(EvalError::TypeMismatch {
                op: op.into(),
                expected: "Logit".into(),
                actual: tag.display_name().into(),
                hint: hint.into(),
            })
        }
    }
}

/// Predicate for ops that consume a Loss (adam, momentum_sgd).
/// Returns Ok if the arg is tagged Loss or has no tag at all;
/// raises TypeMismatch with a tutoring hint otherwise.
pub(crate) fn check_loss_consumer(
    op: &str,
    arg: &Expr,
    env: &Environment,
) -> Result<(), EvalError> {
    match arg_tag(arg, env) {
        None | Some(ValueTag::Loss { .. }) => Ok(()),
        Some(tag) => {
            let hint = match &tag {
                ValueTag::Probability => HINT_LOSS_GOT_PROBABILITY,
                ValueTag::Logit => HINT_LOSS_GOT_LOGIT,
                _ => HINT_LOSS_GOT_OTHER,
            };
            Err(EvalError::TypeMismatch {
                op: op.into(),
                expected: "Loss".into(),
                actual: tag.display_name().into(),
                hint: hint.into(),
            })
        }
    }
}
