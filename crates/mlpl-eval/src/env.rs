//! Evaluation environment (variable bindings).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use mlpl_array::DenseArray;
use mlpl_core::ValueTag;

use crate::error::EvalError;
use crate::experiment::ExperimentRecord;
use crate::grad::OptimizerState;
use crate::interrupt::Interrupt;
use crate::metric_sink::MetricSink;
use crate::model::ModelSpec;
use crate::tokenizer::TokenizerSpec;
use crate::value::Value;

/// Orchestrator hook for remote device peers. Implemented by
/// `mlpl-serve`; absent in local REPL / web eval, where device blocks
/// fall back to the in-process path.
pub trait PeerDispatcher: Send + Sync + std::fmt::Debug {
    fn dispatch_block(
        &self,
        _device: &str,
        _source: &str,
        _bindings: HashMap<String, DenseArray>,
    ) -> Option<Result<Value, EvalError>> {
        None
    }

    fn fetch_tensor(&self, peer: &str, _handle: &str) -> Result<DenseArray, EvalError> {
        Err(EvalError::Unsupported(format!(
            "no peer dispatcher can fetch tensor from {peer}"
        )))
    }
}

/// Variable bindings for evaluation.
#[derive(Clone, Debug, Default)]
pub struct Environment {
    pub(crate) vars: HashMap<String, DenseArray>,
    pub(crate) params: HashSet<String>,
    /// Trainable params that `adam` / `momentum_sgd` skip at the
    /// update step. Gradients still flow through; only the
    /// optimizer update is suppressed. Saga 15 step 001.
    pub(crate) frozen_params: HashSet<String>,
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
    /// Per-tensor device placement. Saga 14 step 005: `to_device`
    /// records the target here keyed by the tensor's variable
    /// name. Missing keys default to `"cpu"`. Params allocated
    /// inside a `device("mlx") { }` block are pre-stamped with
    /// `"mlx"` by the model constructors so `apply(model, X)` can
    /// cross-check placement without the caller needing to remember
    /// to call `to_device` on every param.
    pub(crate) tensor_device: HashMap<String, String>,
    /// One-time-warning flag for the "user asked for MLX but the
    /// mlx feature is not compiled in" fallback path. Set on the
    /// first `device("mlx") { }` entry under that condition; the
    /// warning is emitted at most once per `Environment`.
    pub(crate) mlx_fallback_warned: bool,
    /// Optional remote peer dispatcher installed by the orchestrator
    /// around one eval call. Local eval leaves this unset.
    pub(crate) peer_dispatcher: Option<Arc<dyn PeerDispatcher>>,
    /// Peer-resident tensor handles bound by assignment.
    pub(crate) device_tensors: HashMap<String, Value>,
    /// Saga 23 step 001: optional ValueTag attached per binding
    /// name. Auto-tagged by producer ops in steps 002+; consumed
    /// by predicate-checked consumers, `:describe` / `:vars` /
    /// `:tags`, and trace JSON. Re-binding a name with a new tag
    /// overwrites; clearing is explicit via `clear_tag`. Untyped
    /// bindings keep working unchanged (gradual-typing
    /// additivity rule).
    pub(crate) tags: HashMap<String, ValueTag>,
    /// Bindings that hold a BuiltinRef value (`:add`, `:max`,
    /// `:+`, etc.). Sibling to `vars` / `strings` -- a separate
    /// namespace so user variables can't shadow builtin
    /// references and vice versa. `f = :add` lands here;
    /// `eval_expr(Expr::Ident(f))` looks up here after `vars`
    /// and `strings`. Forward-compatible with first-class
    /// functions: when `Value::Function` lands, this map either
    /// gets absorbed or stays as the "named-builtin" subset.
    pub(crate) builtin_refs: HashMap<String, String>,
    /// Saga 21.5 step 001: optional live-metric sink invoked by
    /// `eval_train` after each iteration with every binding ending
    /// in `_metric` whose value is a scalar. Local REPL evals leave
    /// this `None`; the SSE `/eval_stream` handler installs a
    /// channel-backed sink so the events stream to the client.
    pub(crate) metric_sink: Option<Arc<dyn MetricSink>>,
    /// Saga 21.5 step 003: optional cooperative-cancellation token
    /// checked at loop heads (`for`, `train`, `repeat`) and before
    /// every builtin dispatch. Local REPL evals leave this `None`;
    /// the SSE `/eval_stream` and `/eval` handlers install one per
    /// call so the shared `/cancel` endpoint can trip the bool
    /// from a different thread.
    pub(crate) interrupt: Option<Interrupt>,
    /// Saga 29 step 001: record-valued bindings. Sibling to `vars`
    /// / `strings` / `models` -- a separate namespace so the
    /// existing one-Value-variant-per-map pattern holds.
    /// `r = {X: 1, Y: 2}` lands here; `eval_expr(Expr::Ident("r"))`
    /// looks up here before falling through to other variant maps.
    pub(crate) records: HashMap<String, BTreeMap<String, Value>>,
    /// Saga 29 step 002: string-list-valued bindings. Same
    /// namespace pattern as `records`. `names = ["cat", "dog"]`
    /// lands here.
    pub(crate) string_lists: HashMap<String, Vec<String>>,
    pub(crate) results: HashMap<String, (bool, Value)>, // Saga 29 step 012
    pub(crate) cli_args: Vec<String>,                   // Saga 31 step 003
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

    /// Saga 15 step 001: mark `name` as frozen. `adam` and
    /// `momentum_sgd` skip any frozen name when applying
    /// parameter updates.
    pub fn mark_frozen(&mut self, name: &str) {
        self.frozen_params.insert(name.to_string());
    }

    /// Saga 15 step 001: remove `name` from the frozen set.
    /// No-op if `name` is not currently frozen.
    pub fn unmark_frozen(&mut self, name: &str) {
        self.frozen_params.remove(name);
    }

    /// Saga 15 step 001: whether `name` is currently frozen.
    #[must_use]
    pub fn is_frozen(&self, name: &str) -> bool {
        self.frozen_params.contains(name)
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

    /// Iterate over every bound `(name, ModelSpec)`. Saga 21 step
    /// 002: needed by `mlpl-serve`'s `/inspect` endpoint to list
    /// model names without exposing the internal HashMap.
    pub fn models_iter(&self) -> impl Iterator<Item = (&String, &ModelSpec)> {
        self.models.iter()
    }

    /// Iterate over every bound `(name, TokenizerSpec)`. Saga 21
    /// step 002: same use case as `models_iter`.
    pub fn tokenizers_iter(&self) -> impl Iterator<Item = (&String, &TokenizerSpec)> {
        self.tokenizers.iter()
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

    /// Device placement recorded for the tensor bound to `name`.
    /// Saga 14 step 005. Returns `"cpu"` for names that were never
    /// passed through `to_device` and for names that don't exist
    /// (the evaluator looks up the value separately via `get`).
    #[must_use]
    pub fn tensor_device(&self, name: &str) -> &str {
        self.tensor_device.get(name).map_or("cpu", String::as_str)
    }

    /// Stamp `name` with device placement `target`. Saga 14 step
    /// 005; used by `to_device(x, target)` and by model
    /// constructors when they allocate params inside a
    /// `device("mlx") { }` block.
    pub fn set_tensor_device(&mut self, name: String, target: String) {
        self.tensor_device.insert(name, target);
    }

    pub fn set_peer_dispatcher(&mut self, dispatcher: Arc<dyn PeerDispatcher>) {
        self.peer_dispatcher = Some(dispatcher);
    }

    pub fn clear_peer_dispatcher(&mut self) {
        self.peer_dispatcher = None;
    }

    #[must_use]
    pub fn peer_dispatcher(&self) -> Option<Arc<dyn PeerDispatcher>> {
        self.peer_dispatcher.clone()
    }

    pub fn set_device_tensor(&mut self, name: String, value: Value) {
        self.device_tensors.insert(name, value);
    }

    #[must_use]
    pub fn get_device_tensor(&self, name: &str) -> Option<&Value> {
        self.device_tensors.get(name)
    }

    pub fn remove_device_tensor(&mut self, name: &str) {
        self.device_tensors.remove(name);
    }

    /// Saga 23 step 001: attach a `ValueTag` to a binding name.
    /// Overwrites any prior tag for the same name.
    pub fn set_tag(&mut self, name: String, tag: ValueTag) {
        self.tags.insert(name, tag);
    }

    /// Saga 23 step 001: look up a binding's tag, if any.
    #[must_use]
    pub fn get_tag(&self, name: &str) -> Option<&ValueTag> {
        self.tags.get(name)
    }

    /// Saga 23 step 001: clear any tag attached to `name`. No-op
    /// when `name` is not currently tagged.
    pub fn clear_tag(&mut self, name: &str) {
        self.tags.remove(name);
    }

    /// Saga 23 step 001: iterate every (name, tag) pair. Used by
    /// `:tags` listing in step 005.
    pub fn tags_iter(&self) -> impl Iterator<Item = (&String, &ValueTag)> {
        self.tags.iter()
    }

    /// Bind `name` to a builtin / operator reference (e.g.
    /// `f = :add` records `name="f", target="add"`).
    /// Saga 29 step 001: bind a record value.
    pub fn set_record(&mut self, name: String, fields: BTreeMap<String, Value>) {
        self.records.insert(name, fields);
    }

    /// Saga 29 step 001: look up a record by name.
    #[must_use]
    pub fn get_record(&self, name: &str) -> Option<&BTreeMap<String, Value>> {
        self.records.get(name)
    }

    /// Saga 29 step 002: bind a string-list value.
    pub fn set_string_list(&mut self, name: String, items: Vec<String>) {
        self.string_lists.insert(name, items);
    }

    /// Saga 29 step 002: look up a string-list by name.
    #[must_use]
    pub fn get_string_list(&self, name: &str) -> Option<&Vec<String>> {
        self.string_lists.get(name)
    }

    pub fn set_builtin_ref(&mut self, name: String, target: String) {
        self.builtin_refs.insert(name, target);
    }

    /// Look up a builtin / operator reference by binding name.
    /// Returns the target builtin name, or `None` if `name` is
    /// not bound to a BuiltinRef.
    #[must_use]
    pub fn get_builtin_ref(&self, name: &str) -> Option<&String> {
        self.builtin_refs.get(name)
    }

    /// Saga 21.5 step 001: install a live `MetricSink`. Called by
    /// the SSE `/eval_stream` handler around one eval call;
    /// `clear_metric_sink` removes it on exit.
    pub fn set_metric_sink(&mut self, sink: Arc<dyn MetricSink>) {
        self.metric_sink = Some(sink);
    }

    /// Saga 21.5 step 001: remove any installed `MetricSink`.
    pub fn clear_metric_sink(&mut self) {
        self.metric_sink = None;
    }

    /// Saga 21.5 step 001: borrow the installed `MetricSink`, if
    /// any. `eval_train` clones the `Arc` per iteration so the
    /// emission loop can call out without holding an immutable
    /// borrow of the environment alongside the var iterator.
    #[must_use]
    pub fn metric_sink(&self) -> Option<Arc<dyn MetricSink>> {
        self.metric_sink.clone()
    }

    /// Saga 21.5 step 003: install a cancellation token. The
    /// server's `/cancel` handler flips the same `Arc<AtomicBool>`
    /// from a different thread; `check_interrupt` reads it at
    /// every loop / pre-builtin checkpoint and raises
    /// `EvalError::Cancelled` on trip.
    pub fn set_interrupt(&mut self, interrupt: Interrupt) {
        self.interrupt = Some(interrupt);
    }

    /// Saga 21.5 step 003: drop the installed cancellation token.
    /// Called by the server when an eval call returns so the next
    /// call on the same session starts from a clean slate.
    pub fn clear_interrupt(&mut self) {
        self.interrupt = None;
    }

    /// Saga 21.5 step 003: check the installed cancellation token,
    /// if any. Returns `Err(EvalError::Cancelled { step: 0,
    /// partial_losses: vec![] })` on trip; the enclosing loop
    /// (`eval_train`) re-wraps that error with its own iteration
    /// index + accumulated loss curve before returning.
    pub fn check_interrupt(&self) -> Result<(), EvalError> {
        if self.interrupt.as_ref().is_some_and(Interrupt::is_set) {
            Err(EvalError::Cancelled {
                step: 0,
                partial_losses: Vec::new(),
            })
        } else {
            Ok(())
        }
    }

    /// Saga 21.5 step 001: emit every `_metric`-suffixed scalar
    /// binding for iteration `step` through the installed
    /// `MetricSink`. Called by `eval_train` at the end of each
    /// iteration; no-op when no sink is installed. Extracted
    /// from `eval_train` to keep that function under the
    /// sw-checklist 50-line budget.
    pub(crate) fn emit_metrics(&self, step: usize) {
        let Some(sink) = self.metric_sink() else {
            return;
        };
        let metrics: Vec<(String, f64)> = self
            .vars_iter()
            .filter(|(name, arr)| name.ends_with("_metric") && arr.rank() == 0)
            .map(|(name, arr)| (name.clone(), arr.data()[0]))
            .collect();
        for (name, value) in metrics {
            sink.emit(&name, step, value);
        }
    }
}

/// Public test helper: walk the parameter tree of the model bound to
/// `name` and return the flat list of param identifiers it owns.
#[must_use]
pub fn model_params(env: &Environment, name: &str) -> Option<Vec<String>> {
    env.get_model(name).map(ModelSpec::params)
}
