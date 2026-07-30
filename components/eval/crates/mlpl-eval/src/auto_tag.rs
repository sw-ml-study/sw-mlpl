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

use crate::env_api::*;
use mlpl_core::{ActivationKind, LossKind, ValueTag};
use mlpl_parser::Expr;

use crate::env::Environment;
use mlpl_eval_core::model::{ActKind, ModelSpec};

/// Inspect the right-hand side of an assignment and return the
/// `ValueTag` that producer should auto-attach to the binding,
/// or `None` if no rule matches.
pub(crate) fn for_assign(value: &Expr, env: &Environment) -> Option<ValueTag> {
    match value {
        Expr::FnCall { name, args, .. } if name == "apply" => apply_tag(args, env),
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
    match args.get(1)? {
        Expr::Ident(name, _) => Some(ValueTag::Gradient { wrt: name.clone() }),
        _ => None,
    }
}

/// Auto-tag for `apply(model_ident, X)`. Looks up the model in
/// the environment, walks its `ModelSpec` tail (chain -> last
/// child, residual -> inner), and returns:
/// - `Logit` if the tail is `Linear` or `LinearLora`,
/// - `Probability` if the tail is `Activation(Softmax)`,
/// - `Activation { layer: <model-binding-name>, kind }` for
///   `Activation(Tanh|Relu)`,
/// - `None` otherwise (rms_norm tail, attention tail, embedding
///   tail, non-Ident first arg, unknown model).
fn apply_tag(args: &[Expr], env: &Environment) -> Option<ValueTag> {
    let model_name = match args.first()? {
        Expr::Ident(n, _) => n,
        _ => return None,
    };
    let spec = env.get_model(model_name)?;
    tail_tag(spec, model_name)
}

fn tail_tag(spec: &ModelSpec, layer: &str) -> Option<ValueTag> {
    match spec {
        ModelSpec::Chain(children) => children.last().and_then(|c| tail_tag(c, layer)),
        ModelSpec::Residual(inner) => tail_tag(inner, layer),
        ModelSpec::Linear { .. } | ModelSpec::LinearLora { .. } => Some(ValueTag::Logit),
        ModelSpec::Activation(kind) => activation_tail_tag(*kind, layer),
        ModelSpec::Embedding { .. }
        | ModelSpec::Attention { .. }
        | ModelSpec::RmsNorm { .. }
        | ModelSpec::Engram { .. } => None,
    }
}

/// Saga 23 step 007: derive (input_types, output_type) for a
/// trace event from the expression that produced it. For
/// Assign expressions, the output type is the just-set tag on
/// the binding; input types come from FnCall args that are
/// bare identifiers (tags looked up by name). Other expression
/// kinds yield empty / None. The input_types vec is also
/// emptied when no input carries a tag, so untagged programs
/// serialize identically to their pre-step-007 trace JSON.
pub(crate) fn for_trace_event(
    expr: &Expr,
    env: &Environment,
) -> (Vec<Option<ValueTag>>, Option<ValueTag>) {
    let Expr::Assign { name, value, .. } = expr else {
        return (Vec::new(), None);
    };
    let output_type = env.get_tag(name).cloned();
    let input_types: Vec<Option<ValueTag>> = match value.as_ref() {
        Expr::FnCall { args, .. } => args
            .iter()
            .map(|a| match a {
                Expr::Ident(n, _) => env.get_tag(n).cloned(),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let input_types = if input_types.iter().any(Option::is_some) {
        input_types
    } else {
        Vec::new()
    };
    (input_types, output_type)
}

fn activation_tail_tag(kind: ActKind, layer: &str) -> Option<ValueTag> {
    match kind {
        ActKind::Softmax => Some(ValueTag::Probability),
        ActKind::Tanh => Some(ValueTag::Activation {
            layer: layer.into(),
            kind: ActivationKind::Tanh,
        }),
        ActKind::Relu => Some(ValueTag::Activation {
            layer: layer.into(),
            kind: ActivationKind::Relu,
        }),
    }
}
