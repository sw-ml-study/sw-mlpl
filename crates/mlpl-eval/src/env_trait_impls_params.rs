//! Saga 33 step 010: `impl HasParams for Environment`.

use mlpl_array::DenseArray;
use mlpl_env_traits::HasParams;

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
