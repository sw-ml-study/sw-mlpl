//! Saga 33 step 009: implement the capability traits from
//! `mlpl-env-traits` on `Environment`. Trait method bodies
//! delegate to the inherent methods that the env_* sibling
//! files already provide; this file is the single "plug" that
//! lets downstream sub-crates depend on the traits without
//! ever importing `Environment`.
//!
//! Step 009 seeded with just `HasModels`. As more traits land
//! in `mlpl-env-traits`, their impls land here next to it.

use mlpl_env_traits::HasModels;
use mlpl_eval_core::model::ModelSpec;

use crate::env::Environment;

impl HasModels for Environment {
    fn get_model(&self, name: &str) -> Option<&ModelSpec> {
        Environment::get_model(self, name)
    }
}
