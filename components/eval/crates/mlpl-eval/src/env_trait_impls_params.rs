//! Saga 33 step 010+015: `impl HasParams + HasFrozen for
//! Environment`. Bundled to mirror mlpl-env-traits's
//! params.rs (which holds both traits).

use mlpl_array::DenseArray;
use mlpl_env_traits::{HasFrozen, HasParams};

use crate::env::Environment;

impl HasParams for Environment {
    fn set_param(&mut self, name: String, value: DenseArray) {
        Environment::set_param(self, name, value);
    }
    fn is_param(&self, name: &str) -> bool {
        Environment::is_param(self, name)
    }
    fn mark_param(&mut self, name: &str) {
        Environment::mark_param(self, name);
    }
}

impl HasFrozen for Environment {
    fn mark_frozen(&mut self, name: &str) {
        Environment::mark_frozen(self, name);
    }
    fn unmark_frozen(&mut self, name: &str) {
        Environment::unmark_frozen(self, name);
    }
    fn is_frozen(&self, name: &str) -> bool {
        Environment::is_frozen(self, name)
    }
}
