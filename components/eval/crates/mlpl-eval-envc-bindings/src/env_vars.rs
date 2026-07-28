//! Saga 33 step 002: variable-binding methods extracted from
//! `env.rs`. Rust allows multiple `impl Environment` blocks
//! across modules in the same crate; this file hosts the
//! `vars` `HashMap` accessors. Trainable-parameter logic lives
//! in `env_params.rs`.

use mlpl_array::DenseArray;

use mlpl_eval_env::Environment;

/// Array variable bindings: the core `vars` map accessors.
pub trait EnvVars {
    /// Look up a variable by name.
    #[must_use]
    fn get(&self, name: &str) -> Option<&DenseArray>;
    /// Set a variable binding.
    fn set(&mut self, name: String, value: DenseArray);
    /// Iterate over every bound `(name, DenseArray)`. Used by
    /// `experiment` to scan for `_metric`-suffixed scalars.
    fn vars_iter(&self) -> impl Iterator<Item = (&String, &DenseArray)>;
}

impl EnvVars for Environment {
    fn get(&self, name: &str) -> Option<&DenseArray> {
        self.vars.get(name)
    }

    fn set(&mut self, name: String, value: DenseArray) {
        self.vars.insert(name, value);
    }

    fn vars_iter(&self) -> impl Iterator<Item = (&String, &DenseArray)> {
        self.vars.iter()
    }
}
