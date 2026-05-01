//! Saga 23 step 002: auto-tagging from producer ops.
//!
//! When an `Expr::Assign { name, value }` evaluates, we inspect the
//! right-hand side AST and -- for the curated Tier A producers --
//! attach a `ValueTag` to the binding via `Environment::set_tag`.
//!
//! Step 002 ships the *FnCall-driven* producer rules whose
//! builtins exist at the language surface today: softmax, sigmoid,
//! cross_entropy, cosine_schedule, linear_warmup, attention_weights,
//! and grad.
//!
//! Forward-looking tags whose producer builtins ship in later
//! sagas (LogProbability via `log_softmax`, KLDivergence via
//! `kl_divergence` from Saga 24, etc.) are not wired here -- they
//! land alongside their builtins. The `ValueTag` enum keeps the
//! variants reserved.
//!
//! Three rule clusters that need structural inspection of model
//! values -- Weight/Bias on `linear` / `embed` / `attention` param
//! creation, Logit on `apply` final-layer outputs, and Activation
//! on `apply` through activation layers -- are deferred to step
//! 003 because they require model_dispatch hooks and a structural
//! analysis pass over `ModelSpec` rather than a single AST match.
//!
//! Untyped producers stay untyped: a value with no rule simply
//! gets no tag and continues to flow through the language as a
//! plain DenseArray.

use mlpl_core::{LossKind, ValueTag};
use mlpl_parser::Expr;

/// Inspect the right-hand side of an assignment and return the
/// `ValueTag` that producer should auto-attach to the binding,
/// or `None` if no rule matches.
pub(crate) fn for_assign(value: &Expr) -> Option<ValueTag> {
    match value {
        Expr::FnCall { name, args, .. } => from_fncall(name, args),
        _ => None,
    }
}

fn from_fncall(name: &str, args: &[Expr]) -> Option<ValueTag> {
    match name {
        "softmax" | "sigmoid" => Some(ValueTag::Probability),
        "cross_entropy" => Some(ValueTag::Loss {
            kind: LossKind::CrossEntropy,
        }),
        "cosine_schedule" | "linear_warmup" => Some(ValueTag::LearningRate),
        "attention_weights" => Some(ValueTag::AttentionMap),
        "grad" => gradient_tag(args),
        _ => None,
    }
}

fn gradient_tag(args: &[Expr]) -> Option<ValueTag> {
    // grad(loss_expr, wrt_ident) -- wrt is always an Ident at the
    // language level (the evaluator rejects anything else with a
    // structured error). When the second argument is something
    // else (parser fallthrough or future extensions), we simply
    // skip the tag rather than guess.
    match args.get(1)? {
        Expr::Ident(name, _) => Some(ValueTag::Gradient { wrt: name.clone() }),
        _ => None,
    }
}
