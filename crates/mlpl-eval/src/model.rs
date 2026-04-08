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

/// A model node. For step 001 only the atomic `Linear` layer exists;
/// later steps will add `Chain`, `Activation`, `Residual`, `Norm`,
/// and `Attention` variants.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelSpec {
    /// `linear(in_dim, out_dim, seed)` -- y = X @ W + b, where
    /// `W` has shape `[in_dim, out_dim]` and `b` has shape
    /// `[1, out_dim]`. The two `String` fields are the parameter
    /// names under which `W` and `b` are stored in the environment.
    Linear {
        /// Name of the weight parameter (`[in_dim, out_dim]`).
        w: String,
        /// Name of the bias parameter (`[1, out_dim]`).
        b: String,
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
        }
    }
}
