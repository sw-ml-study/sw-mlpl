//! `mlpl-env-traits` DATA impls for `Environment`: `HasVars`,
//! `HasStrings`, `HasModels` (consolidated from three per-trait files
//! in the eval decomposition's spine tidy).

use mlpl_array::DenseArray;
use mlpl_env_traits::HasModels;
use mlpl_env_traits::HasStrings;
use mlpl_env_traits::HasVars;
use mlpl_eval_core::model::ModelSpec;

use crate::env::Environment;

impl HasVars for Environment {
    fn get(&self, name: &str) -> Option<&DenseArray> {
        self.vars.get(name)
    }
    fn set(&mut self, name: String, value: DenseArray) {
        self.vars.insert(name, value);
    }
}

impl HasStrings for Environment {
    fn set_string(&mut self, name: String, value: String) {
        self.strings.insert(name, value);
    }
    fn get_string(&self, name: &str) -> Option<&String> {
        self.strings.get(name)
    }
}

impl HasModels for Environment {
    fn get_model(&self, name: &str) -> Option<&ModelSpec> {
        self.models.get(name)
    }
}
