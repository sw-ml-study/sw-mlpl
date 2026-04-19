//! Evaluation environment (variable bindings).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use mlpl_array::DenseArray;

use crate::experiment::ExperimentRecord;
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
    /// String-valued variable bindings (Saga 12 step 009). Sibling
    /// to `vars`; populated when assignment value is a `Value::Str`
    /// (e.g. `corpus = load_preloaded("...")`). Ident lookup checks
    /// here before falling through to array vars.
    pub(crate) strings: HashMap<String, String>,
    /// Output directory for `experiment` records. `None` disables
    /// disk writes (web REPL); `Some(path)` is set by the terminal
    /// REPL's `--exp-dir` flag. Saga 12 step 007.
    pub(crate) exp_dir: Option<PathBuf>,
    /// Append-only log of completed experiments in this env.
    pub(crate) experiment_log: Vec<ExperimentRecord>,
    /// Stack of active `device("...")` targets. Empty stack means
    /// the implicit CPU device. The innermost (last) entry wins,
    /// so nested `device("mlx") { device("cpu") { ... } }` runs
    /// the inner body on CPU even when the outer is MLX. Saga 14
    /// step 004.
    pub(crate) device_stack: Vec<String>,
    /// One-time-warning flag for the "user asked for MLX but the
    /// mlx feature is not compiled in" fallback path. Set on the
    /// first `device("mlx") { }` entry under that condition; the
    /// warning is emitted at most once per `Environment`.
    pub(crate) mlx_fallback_warned: bool,
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

    /// Set the output directory for `experiment` records. Saga 12
    /// step 007. The terminal REPL calls this from a `--exp-dir`
    /// CLI flag; the web REPL leaves it unset so nothing is
    /// written to disk.
    pub fn set_exp_dir(&mut self, dir: PathBuf) {
        self.exp_dir = Some(dir);
    }

    /// Borrow the configured experiment output dir, if any.
    #[must_use]
    pub fn exp_dir(&self) -> Option<&PathBuf> {
        self.exp_dir.as_ref()
    }

    /// Append a completed experiment record to the log.
    pub fn push_experiment_log(&mut self, rec: ExperimentRecord) {
        self.experiment_log.push(rec);
    }

    /// Borrow every recorded experiment in order.
    #[must_use]
    pub fn experiment_log(&self) -> &[ExperimentRecord] {
        &self.experiment_log
    }

    /// Iterate over every bound `(name, DenseArray)`. Used by
    /// `experiment` to scan for `_metric`-suffixed scalars.
    pub fn vars_iter(&self) -> impl Iterator<Item = (&String, &DenseArray)> {
        self.vars.iter()
    }

    /// Bind a string value to `name`. Saga 12 step 009.
    pub fn set_string(&mut self, name: String, value: String) {
        self.strings.insert(name, value);
    }

    /// Look up a string binding by name.
    #[must_use]
    pub fn get_string(&self, name: &str) -> Option<&String> {
        self.strings.get(name)
    }

    /// Current active device target (Saga 14 step 004). Returns
    /// `"cpu"` when no `device("...")` block is in scope.
    #[must_use]
    pub fn device(&self) -> &str {
        self.device_stack.last().map_or("cpu", String::as_str)
    }

    /// Push a new device target onto the stack. Called on
    /// `device("...") { ... }` entry.
    pub fn push_device(&mut self, target: String) {
        self.device_stack.push(target);
    }

    /// Pop the innermost device target. Called on `device(...)`
    /// block exit. No-op when the stack is empty (defensive).
    pub fn pop_device(&mut self) {
        self.device_stack.pop();
    }

    /// Take ownership of the "have we already warned about an MLX
    /// fallback?" flag. Returns `true` the first time it is
    /// called per `Environment`, `false` thereafter, so callers
    /// can emit a warning at most once.
    pub fn take_mlx_fallback_warning(&mut self) -> bool {
        if self.mlx_fallback_warned {
            return false;
        }
        self.mlx_fallback_warned = true;
        true
    }
}

/// Public test helper: walk the parameter tree of the model bound to
/// `name` and return the flat list of param identifiers it owns.
#[must_use]
pub fn model_params(env: &Environment, name: &str) -> Option<Vec<String>> {
    env.get_model(name).map(ModelSpec::params)
}
