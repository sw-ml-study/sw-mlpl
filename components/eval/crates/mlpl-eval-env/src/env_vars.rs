//! Saga 33 step 002: variable-binding methods extracted from
//! `env.rs`. Rust allows multiple `impl Environment` blocks
//! across modules in the same crate; this file hosts the
//! `vars` `HashMap` accessors. Trainable-parameter logic lives
//! in `env_params.rs`.

use mlpl_array::DenseArray;

use crate::env::Environment;

impl Environment {
    /// Look up a variable by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&DenseArray> {
        self.vars.get(name)
    }

    /// Set a variable binding.
    pub fn set(&mut self, name: String, value: DenseArray) {
        self.vars.insert(name, value);
    }

    /// Iterate over every bound `(name, DenseArray)`. Used by
    /// `experiment` to scan for `_metric`-suffixed scalars.
    pub fn vars_iter(&self) -> impl Iterator<Item = (&String, &DenseArray)> {
        self.vars.iter()
    }
}
