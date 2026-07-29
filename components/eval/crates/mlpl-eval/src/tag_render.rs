//! Saga 23 step 006: per-tag rendering helpers for `:describe`,
//! `:vars`, and the `:tags` listing.
//!
//! The bodies below render the curated Tier A vocabulary into
//! short ASCII text. Educational-first: every tag's description
//! should let a student understand why the tag exists and what
//! op produced or consumes it.

use crate::env_api::*;
use mlpl_array::DenseArray;
use mlpl_core::{ActivationKind, LossKind, ValueTag};

use crate::env::Environment;

/// One-line tag header for `:describe`.
#[must_use]
pub(crate) fn header_line(tag: &ValueTag) -> String {
    match tag {
        ValueTag::Logit | ValueTag::Probability | ValueTag::LogProbability => {
            tag.display_name().into()
        }
        ValueTag::Loss { kind } => format!("Loss({})", loss_kind_str(*kind)),
        ValueTag::Gradient { wrt } => format!("Gradient(wrt={wrt})"),
        ValueTag::Weight { layer, name } => {
            format!("Weight(layer={layer}, name={name})")
        }
        ValueTag::Bias { layer } => format!("Bias(layer={layer})"),
        ValueTag::Activation { layer, kind } => format!(
            "Activation(layer={layer}, kind={})",
            activation_kind_str(*kind)
        ),
        ValueTag::LearningRate => "LearningRate".into(),
        ValueTag::Labels { num_classes } => format!("Labels(num_classes={num_classes})"),
        ValueTag::AttentionMap => "AttentionMap".into(),
    }
}

/// Per-tag body lines for `:describe`. Probability includes a
/// row-sum invariant check; AttentionMap notes the heatmap
/// rendering; Loss / Gradient / Activation reiterate metadata.
pub(crate) fn body_lines(tag: &ValueTag, arr: Option<&DenseArray>) -> Vec<String> {
    match tag {
        ValueTag::Probability => arr
            .map(|a| vec![probability_invariant(a)])
            .unwrap_or_default(),
        ValueTag::AttentionMap => {
            vec!["renders as a heatmap via svg(att, \"heatmap\")".into()]
        }
        ValueTag::Loss { kind } => vec![format!(
            "scalar loss; produced by the {} family",
            loss_kind_op(*kind)
        )],
        ValueTag::Gradient { wrt } => vec![format!(
            "gradient w.r.t. parameter `{wrt}` (consumed by adam / momentum_sgd)"
        )],
        ValueTag::Weight { .. } => vec!["trainable parameter (matrix)".into()],
        ValueTag::Bias { .. } => vec!["trainable parameter (bias row)".into()],
        ValueTag::Activation { kind, .. } => vec![format!(
            "hidden activation, kind={}",
            activation_kind_str(*kind)
        )],
        ValueTag::Labels { num_classes } => vec![format!(
            "integer-valued class indices in [0, {num_classes})"
        )],
        ValueTag::Logit => vec!["unnormalized scores; bridge with softmax(_, axis)".into()],
        ValueTag::LogProbability => {
            vec!["log-probabilities; bridge with exp() to recover Probability".into()]
        }
        ValueTag::LearningRate => Vec::new(),
    }
}

/// `:tags` listing. Walks `Environment::tags`, sorts by name,
/// and renders one line per binding using `header_line`.
pub(crate) fn format_tags(env: &Environment) -> String {
    let mut entries: Vec<(&String, &ValueTag)> = env.tags_iter().collect();
    if entries.is_empty() {
        return "(no tagged bindings)".into();
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    for (name, tag) in entries {
        out.push_str(&format!("  {name}: {}\n", header_line(tag)));
    }
    out.truncate(out.trim_end().len());
    out
}

fn loss_kind_str(k: LossKind) -> &'static str {
    match k {
        LossKind::CrossEntropy => "CrossEntropy",
        LossKind::Mse => "Mse",
        LossKind::KlDivergence => "KlDivergence",
        LossKind::Custom => "Custom",
    }
}

fn loss_kind_op(k: LossKind) -> &'static str {
    match k {
        LossKind::CrossEntropy => "cross_entropy",
        LossKind::Mse => "mse",
        LossKind::KlDivergence => "kl_divergence",
        LossKind::Custom => "custom",
    }
}

fn activation_kind_str(k: ActivationKind) -> &'static str {
    match k {
        ActivationKind::Tanh => "Tanh",
        ActivationKind::Relu => "Relu",
        ActivationKind::Sigmoid => "Sigmoid",
        ActivationKind::Softmax => "Softmax",
    }
}

fn probability_invariant(arr: &DenseArray) -> String {
    let dims = arr.shape().dims();
    if dims.is_empty() {
        return "row-sum invariant: scalar (skipped)".into();
    }
    let last = *dims.last().unwrap();
    if last == 0 {
        return "row-sum invariant: empty (skipped)".into();
    }
    let data = arr.data();
    let max_dev = (0..data.len() / last)
        .map(|r| (data[r * last..(r + 1) * last].iter().sum::<f64>() - 1.0).abs())
        .fold(0.0f64, f64::max);
    if max_dev <= 1e-5 {
        format!("row-sum invariant: verified (max deviation {max_dev:.2e})")
    } else {
        format!("row-sum invariant: violated (max deviation {max_dev:.2e}; rows do not sum to 1)")
    }
}
