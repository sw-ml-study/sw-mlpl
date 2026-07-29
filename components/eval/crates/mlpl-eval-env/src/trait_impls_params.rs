//! Saga 33 step 010+015: `impl HasParams + HasFrozen for
//! Environment`. Bundled to mirror mlpl-env-traits's
//! params.rs (which holds both traits).

use mlpl_array::DenseArray;
use mlpl_env_traits::{HasFrozen, HasParams};

use crate::env::Environment;

impl HasParams for Environment {
    fn set_param(&mut self, name: String, value: DenseArray) {
        self.params.insert(name.clone());
        self.vars.insert(name, value);
    }
    fn is_param(&self, name: &str) -> bool {
        self.params.contains(name)
    }
    fn mark_param(&mut self, name: &str) {
        self.params.insert(name.to_string());
    }
}

impl HasFrozen for Environment {
    fn mark_frozen(&mut self, name: &str) {
        self.frozen_params.insert(name.to_string());
    }
    fn unmark_frozen(&mut self, name: &str) {
        self.frozen_params.remove(name);
    }
    fn is_frozen(&self, name: &str) -> bool {
        self.frozen_params.contains(name)
    }
}
