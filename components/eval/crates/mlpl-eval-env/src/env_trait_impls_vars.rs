//! Saga 33 step 012: `impl HasVars for Environment`.

use mlpl_array::DenseArray;
use mlpl_env_traits::HasVars;

use crate::env::Environment;

impl HasVars for Environment {
    fn get(&self, name: &str) -> Option<&DenseArray> {
        self.vars.get(name)
    }
    fn set(&mut self, name: String, value: DenseArray) {
        self.vars.insert(name, value);
    }
}
