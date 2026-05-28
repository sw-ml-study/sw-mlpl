//! Saga 33 step 003: model-registry methods extracted from
//! `env.rs`. Tokenizer registry lives in the sibling
//! `env_tokenizers.rs`. Each is a thin `HashMap` accessor;
//! the actual `ModelSpec` lives in `mlpl-eval-core::model`.

use crate::env::Environment;
use mlpl_eval_core::model::ModelSpec;

impl Environment {
    /// Look up a model by name (Saga 11). Returns `None` if `name`
    /// is not bound to a model value.
    #[must_use]
    pub fn get_model(&self, name: &str) -> Option<&ModelSpec> {
        self.models.get(name)
    }

    /// Iterate over every bound `(name, ModelSpec)`. Saga 21 step
    /// 002: needed by `mlpl-serve`'s `/inspect` endpoint to list
    /// model names without exposing the internal HashMap.
    pub fn models_iter(&self) -> impl Iterator<Item = (&String, &ModelSpec)> {
        self.models.iter()
    }
}
