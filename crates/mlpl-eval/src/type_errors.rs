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

use mlpl_core::ValueTag;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;

const HINT_LOGIT_GOT_PROBABILITY: &str = "\
cross_entropy / sample / top_k expect raw scores (Logit), not \
probabilities. you applied softmax already. fix: pass the \
original Logit, or for cross_entropy switch to nll(probs, y) \
once Saga 24 ships nll. inside cross_entropy specifically, \
double-softmax produces a NaN factory at scale.";

const HINT_LOGIT_GOT_LOG_PROBABILITY: &str = "\
this consumer expects raw scores (Logit), but the argument is \
already log-softmaxed. fix: pass the original Logit. for \
cross_entropy, the equivalent on log-probs is nll(log_probs, y) \
which Saga 24 will ship.";

const HINT_LOSS_GOT_PROBABILITY: &str = "\
adam / momentum_sgd expect a scalar Loss, but the argument is a \
Probability. did you forget the loss function? fix: wrap the \
probabilities in a loss term, e.g. cross_entropy(logits, y) or \
mse(probs, target).";

const HINT_LOSS_GOT_LOGIT: &str = "\
adam / momentum_sgd expect a scalar Loss, but the argument is a \
Logit. fix: pass the loss term, not the predictions, e.g. \
loss = cross_entropy(logits, y); adam(loss, params, ...).";

const HINT_LOSS_GOT_OTHER: &str = "\
adam / momentum_sgd expect a scalar Loss-tagged value. produce \
one with cross_entropy / mse / kl_divergence and pass that as \
the first argument.";

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
