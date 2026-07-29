//! Saga 33 step 009: `impl HasModels for Environment`. Single
//! trait method delegating to the inherent `Environment::
//! get_model` defined in `env_models.rs`.

use mlpl_env_traits::HasModels;
use mlpl_eval_core::model::ModelSpec;

use crate::env::Environment;

impl HasModels for Environment {
    fn get_model(&self, name: &str) -> Option<&ModelSpec> {
        self.models.get(name)
    }
}
