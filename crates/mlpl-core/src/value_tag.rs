//! Saga 23 step 001: ValueTag side-table machinery.
//!
//! `ValueTag` is the curated Tier A vocabulary from
//! `docs/optional-typing-design.md`. Tags attach to bindings via the
//! `Environment::tags` side table (defined in `mlpl-eval`); this
//! module owns the tag enum and its helper kinds so that downstream
//! crates (`mlpl-trace`, `mlpl-cli`) can serialize tags without
//! depending on `mlpl-eval`.
//!
//! No language surface lands in this step. Auto-tagging from
//! producer ops (softmax, grad, linear, cross_entropy, ...) and
//! predicate-checked consumers ship in steps 002 and 003 of this
//! saga.

use serde::{Deserialize, Serialize};

/// Loss-family discriminator carried by `ValueTag::Loss`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossKind {
    CrossEntropy,
    Mse,
    KlDivergence,
    Custom,
}

/// Activation-family discriminator carried by `ValueTag::Activation`.
///
/// Mirrors `mlpl_eval::model::ActKind` deliberately: that enum is
/// runtime model-spec metadata; this enum is the pedagogical tag
/// surface exposed in `:describe` and trace JSON. They stay
/// separate so the typing layer remains independent of the model
/// DSL crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationKind {
    Tanh,
    Relu,
    Sigmoid,
    Softmax,
}

/// The Tier A typed-value vocabulary.
///
/// Tags are *predicates* the runtime can attach to a binding name
/// and check at op boundaries (Saga 23 steps 002+). Untyped values
/// remain first-class -- `set_tag` is opt-in and any consumer that
/// gets an untagged value passes the predicate by default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueTag {
    /// Unnormalized scores. Produced by a model's final `Linear`.
    /// Consumed by `cross_entropy`, `sample`, `top_k`.
    Logit,
    /// Probabilities (per-row sum 1.0). Produced by `softmax`,
    /// `sigmoid`. Consumed by `nll`, `entropy`, `kl_divergence`.
    Probability,
    /// Log-probabilities. Produced by `log_softmax`. Accepted by
    /// `nll`; rejected by `cross_entropy`.
    LogProbability,
    /// Scalar loss. Produced by `cross_entropy`, `mse`,
    /// `kl_divergence`. Consumed by `grad`, `adam`, `momentum_sgd`.
    Loss { kind: LossKind },
    /// Gradient with respect to a named parameter. Produced by
    /// `grad(loss, wrt)`.
    Gradient { wrt: String },
    /// Trainable weight matrix owned by a layer. Produced by
    /// `linear`, `embed`, `attention` constructors.
    Weight { layer: String, name: String },
    /// Trainable bias row owned by a layer. Produced by `linear`,
    /// `embed` constructors.
    Bias { layer: String },
    /// Hidden activation flowing out of a parameter-free layer.
    /// Produced by `tanh_layer`, `relu_layer`, `sigmoid`,
    /// `softmax_layer` apply outputs.
    Activation { layer: String, kind: ActivationKind },
    /// Scalar learning rate from a schedule. Produced by
    /// `cosine_schedule`, `linear_warmup`.
    LearningRate,
    /// Integer-valued label vector for a classification task.
    /// Carries `num_classes` so `confusion_matrix` and
    /// `cross_entropy` can validate compatibility.
    Labels { num_classes: usize },
    /// Per-head per-position attention weights. Produced by
    /// `attention_weights`. Renders as a heatmap in
    /// `svg(_, "heatmap")`.
    AttentionMap,
}

impl ValueTag {
    /// Stable short identifier used by `:describe`, `:tags`, and
    /// trace JSON for the variant kind. Discriminator only -- it
    /// does not include the variant's metadata.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            ValueTag::Logit => "Logit",
            ValueTag::Probability => "Probability",
            ValueTag::LogProbability => "LogProbability",
            ValueTag::Loss { .. } => "Loss",
            ValueTag::Gradient { .. } => "Gradient",
            ValueTag::Weight { .. } => "Weight",
            ValueTag::Bias { .. } => "Bias",
            ValueTag::Activation { .. } => "Activation",
            ValueTag::LearningRate => "LearningRate",
            ValueTag::Labels { .. } => "Labels",
            ValueTag::AttentionMap => "AttentionMap",
        }
    }
}
