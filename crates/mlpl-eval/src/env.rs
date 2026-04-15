//! Evaluation environment (variable bindings).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use mlpl_array::DenseArray;

use crate::grad::OptimizerState;
use crate::model::ModelSpec;
use crate::tokenizer::TokenizerSpec;

/// Variable bindings for evaluation.
#[derive(Clone, Debug, Default)]
pub struct Environment {
    pub(crate) vars: HashMap<String, DenseArray>,
    pub(crate) params: HashSet<String>,
    pub(crate) optim_state: OptimizerState,
    pub(crate) models: HashMap<String, ModelSpec>,
    pub(crate) next_model_id: u64,
    /// Tokenizer bindings (Saga 12 step 004). Sibling to `models`.
    pub(crate) tokenizers: HashMap<String, TokenizerSpec>,
    /// Sandbox root for filesystem `load("relative-path")` calls.
    /// `None` means filesystem access is disabled (the web REPL
    /// surface, where `std::fs` doesn't exist under WASM). Saga 12
    /// step 001.
    pub(crate) data_dir: Option<PathBuf>,
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

    /// Look up a model by name (Saga 11). Returns `None` if `name`
    /// is not bound to a model value.
    #[must_use]
    pub fn get_model(&self, name: &str) -> Option<&ModelSpec> {
        self.models.get(name)
    }

    /// Set the sandbox root for `load("relative-path")` (Saga 12
    /// step 001). The terminal REPL calls this from a `--data-dir`
    /// CLI flag; the web REPL never calls this, leaving fs access
    /// disabled.
    pub fn set_data_dir(&mut self, dir: PathBuf) {
        self.data_dir = Some(dir);
    }

    /// Bind `name` to a tokenizer value. Saga 12 step 004.
    pub fn set_tokenizer(&mut self, name: String, tok: TokenizerSpec) {
        self.tokenizers.insert(name, tok);
    }

    /// Look up a tokenizer by name. Returns `None` if `name` is
    /// not bound to a tokenizer.
    #[must_use]
    pub fn get_tokenizer(&self, name: &str) -> Option<&TokenizerSpec> {
        self.tokenizers.get(name)
    }

    /// Borrow the current sandbox root, if any.
    #[must_use]
    pub fn data_dir(&self) -> Option<&PathBuf> {
        self.data_dir.as_ref()
    }
}

/// Public test helper: walk the parameter tree of the model bound to
/// `name` and return the flat list of param identifiers it owns.
#[must_use]
pub fn model_params(env: &Environment, name: &str) -> Option<Vec<String>> {
    env.get_model(name).map(ModelSpec::params)
}
