//! Saga 33 step 012: `impl HasVars for Environment`.

use mlpl_array::DenseArray;
use mlpl_env_traits::HasVars;

use crate::env::Environment;

impl HasVars for Environment {
    fn get(&self, name: &str) -> Option<&DenseArray> {
        Environment::get(self, name)
    }
    fn set(&mut self, name: String, value: DenseArray) {
        Environment::set(self, name, value);
    }
}
