//! `HasModels`: read-only access to the registered model
//! catalog. The first capability trait extracted from
//! `Environment`. Downstream consumers (e.g. future
//! `mlpl-models-apply`, `mlpl-models-analyze`) take
//! `&impl HasModels` instead of `&Environment` so they can be
//! tested against a minimal Mock that only impls this one
//! trait.

use mlpl_eval_core::model::ModelSpec;

pub trait HasModels {
    /// Look up a model by name. Returns `None` if the name is
    /// not bound to a `ModelSpec` value.
    fn get_model(&self, name: &str) -> Option<&ModelSpec>;
}
