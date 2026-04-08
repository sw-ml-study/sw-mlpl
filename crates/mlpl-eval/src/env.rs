//! Evaluation environment (variable bindings).

use std::collections::{HashMap, HashSet};

use mlpl_array::DenseArray;

use crate::grad::OptimizerState;

/// Variable bindings for evaluation.
#[derive(Clone, Debug, Default)]
pub struct Environment {
    vars: HashMap<String, DenseArray>,
    params: HashSet<String>,
    pub(crate) optim_state: OptimizerState,
}

impl Environment {
    /// Create an empty environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a variable by name.
    pub fn get(&self, name: &str) -> Option<&DenseArray> {
        self.vars.get(name)
    }

    /// Set a variable binding.
    pub fn set(&mut self, name: String, value: DenseArray) {
        self.vars.insert(name, value);
    }

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
