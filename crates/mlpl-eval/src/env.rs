//! Evaluation environment (variable bindings).

use std::collections::HashMap;

use mlpl_array::DenseArray;

/// Variable bindings for evaluation.
#[derive(Clone, Debug, Default)]
pub struct Environment {
    vars: HashMap<String, DenseArray>,
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
}
