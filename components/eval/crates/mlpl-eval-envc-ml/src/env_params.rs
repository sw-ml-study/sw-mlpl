//! Saga 33 step 002: trainable-parameter methods extracted
//! from `env.rs`. The `params` `HashSet` tracks names that
//! `grad` / `adam` / `momentum_sgd` treat as gradient
//! sources. Freezing logic (Saga 15 step 001) lives in the
//! sibling `env_frozen.rs`.

use mlpl_array::DenseArray;

use mlpl_eval_env::Environment;

/// Trainable-parameter tracking.
pub trait EnvParams {
    /// Set a variable and mark it as a trainable parameter (tracked by `grad`).
    fn set_param(&mut self, name: String, value: DenseArray);
    /// Mark an existing variable as a trainable parameter.
    fn mark_param(&mut self, name: &str);
    /// Whether `name` is a trainable parameter in this environment.
    #[must_use]
    fn is_param(&self, name: &str) -> bool;
}

impl EnvParams for Environment {
    fn set_param(&mut self, name: String, value: DenseArray) {
        self.params.insert(name.clone());
        self.vars.insert(name, value);
    }

    fn mark_param(&mut self, name: &str) {
        self.params.insert(name.to_string());
    }

    fn is_param(&self, name: &str) -> bool {
        self.params.contains(name)
    }
}
