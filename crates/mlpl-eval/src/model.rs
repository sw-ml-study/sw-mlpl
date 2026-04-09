//! Model DSL value type (Saga 11).
//!
//! A `ModelSpec` describes a callable layer (or composition of layers)
//! along with the parameter identifiers it owns. Parameters live in
//! the regular `Environment` so they can be referenced by `grad`,
//! `momentum_sgd`, and `adam` exactly like any other tracked param.
//!
//! Apply surface: `apply(model, X)` is the only way to evaluate a
//! model on an input. The `model(X)` function-call form would
//! require knowing at parse time that `model` is a model identifier,
//! which the parser does not know -- a free-standing `apply` built-in
//! sidesteps the symbol-table problem and keeps the surface uniform.

/// Activation kind for the parameter-free activation layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActKind {
    /// Element-wise hyperbolic tangent.
    Tanh,
    /// Element-wise rectified linear unit (`max(x, 0)`).
    Relu,
    /// Row-wise softmax (axis 1).
    Softmax,
}

/// A model node. Built up step by step over Saga 11; later steps add
/// `Residual`, `Norm`, and `Attention` variants.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelSpec {
    /// `linear(in_dim, out_dim, seed)` -- y = X @ W + b.
    Linear {
        /// Name of the weight parameter (`[in_dim, out_dim]`).
        w: String,
        /// Name of the bias parameter (`[1, out_dim]`).
        b: String,
    },
    /// `chain(layer_a, layer_b, ...)` -- sequential composition.
    /// Apply threads the input through each child in order.
    Chain(Vec<ModelSpec>),
    /// Parameter-free activation layer (`tanh_layer`, `relu_layer`,
    /// `softmax_layer`).
    Activation(ActKind),
    /// `residual(inner)` -- y = x + inner(x). The inner model's
    /// output shape must match its input shape.
    Residual(Box<ModelSpec>),
    /// `rms_norm(dim)` -- per-row root-mean-square normalization
    /// (no learnable scale or shift). `dim` records the expected
    /// last-dim size for documentation only; the implementation
    /// normalizes whatever rank-2 input it receives.
    RmsNorm {
        /// Expected last-dim size (informational).
        dim: usize,
    },
}

impl ModelSpec {
    /// Walk the model and return a flat, order-stable list of the
    /// parameter identifiers it owns. Used by `params(model)` and
    /// by tests that need to inspect parameter shapes.
    #[must_use]
    pub fn params(&self) -> Vec<String> {
        match self {
            Self::Linear { w, b } => vec![w.clone(), b.clone()],
            Self::Chain(children) => {
                let mut out = Vec::new();
                for child in children {
                    out.extend(child.params());
                }
                out
            }
            Self::Activation(_) => Vec::new(),
            Self::Residual(inner) => inner.params(),
            Self::RmsNorm { .. } => Vec::new(),
        }
    }
}
