//! Saga 33 step 002: trainable-parameter methods extracted
//! from `env.rs`. The `params` `HashSet` tracks names that
//! `grad` / `adam` / `momentum_sgd` treat as gradient
//! sources. Freezing logic (Saga 15 step 001) lives in the
//! sibling `env_frozen.rs`.

use mlpl_array::DenseArray;

use crate::env::Environment;

impl Environment {
    /// Set a variable and mark it as a trainable parameter (tracked by `grad`).
    pub fn set_param(&mut self, name: String, value: DenseArray) {
        self.params.insert(name.clone());
        self.vars.insert(name, value);
    }

    /// Mark an existing variable as a trainable parameter.
    pub fn mark_param(&mut self, name: &str) {
        self.params.insert(name.to_string());
    }

    /// Whether `name` is a trainable parameter in this environment.
    #[must_use]
    pub fn is_param(&self, name: &str) -> bool {
        self.params.contains(name)
    }

    /// Iterate over all (name, value) parameter bindings.
    pub fn params(&self) -> impl Iterator<Item = (&String, &DenseArray)> {
        self.params
            .iter()
            .filter_map(move |n| self.vars.get(n).map(|v| (n, v)))
    }
}
