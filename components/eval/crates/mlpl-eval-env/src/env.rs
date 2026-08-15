//! Evaluation environment (variable bindings).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use mlpl_array::DenseArray;
use mlpl_core::ValueTag;

use mlpl_eval_core::TokenizerSpec;
use mlpl_eval_core::metric_sink::MetricSink;
use mlpl_eval_core::model::ModelSpec;

/// Reserved `extension_id` marking a `Value::ExtHandle` as a native
/// PORT (vs a real extension handle). Ports are minted on open, so
/// MLPL cannot forge one.
pub const PORT_EXTENSION_ID: u64 = u64::MAX;
/// The port resource kind (`type_id`).
pub const PORT_TYPE_ID: u64 = 1;

/// One port's channel endpoints, owned by the interpreter: the command
/// side (MLPL -> far end) is a `Sender`, the event side (far end ->
/// MLPL) a `Receiver`. Share-nothing -- only owned `Value`s cross.
#[derive(Debug)]
pub struct PortEndpoints {
    pub commands: Sender<Value>,
    pub events: Receiver<Value>,
}
use mlpl_eval_state::ExperimentRecord;
use mlpl_eval_state::Interrupt;
use mlpl_eval_state::OptimizerState;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

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

    /// # Errors
    /// Always errors: no peer dispatcher is wired on this env.
    fn fetch_tensor(&self, peer: &str, _handle: &str) -> Result<DenseArray, EvalError> {
        Err(EvalError::Unsupported(format!(
            "no peer dispatcher can fetch tensor from {peer}"
        )))
    }
}

/// Variable bindings for evaluation.
///
/// Not `Clone`: an `Environment` can own port `Receiver`s (see
/// `ports`), which cannot be cloned -- and cloning a live event queue
/// would be meaningless anyway. Per-frame scope isolation clones
/// individual tables (`env_scope`), never the whole environment.
#[derive(Debug, Default)]
pub struct Environment {
    pub vars: HashMap<String, DenseArray>,
    pub params: HashSet<String>,
    /// Trainable params that `adam` / `momentum_sgd` skip at the
    /// update step. Gradients still flow through; only the
    /// optimizer update is suppressed. Saga 15 step 001.
    pub frozen_params: HashSet<String>,
    pub optim_state: OptimizerState,
    pub models: HashMap<String, ModelSpec>,
    pub next_model_id: u64,
    /// Tokenizer bindings (Saga 12 step 004). Sibling to `models`.
    pub tokenizers: HashMap<String, TokenizerSpec>,
    /// Generation states (KV caches) by binding name -- the
    /// `gen_*` family (docs/kv-cache-design.md).
    pub gen_states: std::collections::HashMap<String, mlpl_eval_core::GenState>,
    /// Partial applications by binding name (name, arity, bound).
    pub partials: std::collections::HashMap<String, (String, usize, Vec<Value>)>,
    /// Registered `test_event_sink` callback (`u:` name), the
    /// typed test-event API's delivery target.
    pub test_event_sink: Option<String>,
    /// Script-mode host transport: append each validated test
    /// event as one JSON line to this path.
    pub test_events_out: Option<std::path::PathBuf>,
    /// Connect-mode host transport: buffered JSON lines the
    /// server drains per eval.
    pub test_event_lines: Option<Vec<String>>,
    /// Explicit global writes recorded inside user-fn frames
    /// (`global_set`); replayed after each frame restore so they
    /// survive to the top level. Cleared when call depth
    /// returns to zero.
    pub global_writes: Vec<(String, Value)>,
    /// Current user-fn call depth (frame replay bookkeeping).
    pub call_depth: usize,
    /// Sandbox root for the filesystem builtins (`fs_walk` /
    /// `read_text` / `write_text` / `remove_path`). None on
    /// surfaces without filesystem access.
    pub fs_root: Option<std::path::PathBuf>,
    /// Host hook for `run_script` (registered by surfaces that
    /// own a source loader; absent elsewhere).
    pub run_script_hook: Option<RunScriptHook>,
    /// When set (`run_script` / tests), `exit(code)` raises the
    /// `ExitRequested` signal instead of terminating the
    /// process -- the dependency-injection seam for exit.
    pub exit_intercept: bool,
    /// Sandbox root for filesystem `load("relative-path")` calls.
    /// `None` means filesystem access is disabled (the web REPL
    /// surface, where `std::fs` doesn't exist under WASM). Saga 12
    /// step 001.
    pub data_dir: Option<PathBuf>,
    /// String-valued variable bindings (Saga 12 step 009). Sibling
    /// to `vars`; populated when assignment value is a `Value::Str`
    /// (e.g. `corpus = load_preloaded("...")`). Ident lookup checks
    /// here before falling through to array vars.
    pub strings: HashMap<String, String>,
    /// Output directory for `experiment` records. `None` disables
    /// disk writes (web REPL); `Some(path)` is set by the terminal
    /// REPL's `--exp-dir` flag. Saga 12 step 007.
    pub exp_dir: Option<PathBuf>,
    /// Append-only log of completed experiments in this env.
    pub experiment_log: Vec<ExperimentRecord>,
    /// Stack of active `device("...")` targets. Empty stack means
    /// the implicit CPU device. The innermost (last) entry wins,
    /// so nested `device("mlx") { device("cpu") { ... } }` runs
    /// the inner body on CPU even when the outer is MLX. Saga 14
    /// step 004.
    pub device_stack: Vec<String>,
    /// Per-tensor device placement. Saga 14 step 005: `to_device`
    /// records the target here keyed by the tensor's variable
    /// name. Missing keys default to `"cpu"`. Params allocated
    /// inside a `device("mlx") { }` block are pre-stamped with
    /// `"mlx"` by the model constructors so `apply(model, X)` can
    /// cross-check placement without the caller needing to remember
    /// to call `to_device` on every param.
    pub tensor_device: HashMap<String, String>,
    /// One-time-warning flag for the "user asked for MLX but the
    /// mlx feature is not compiled in" fallback path. Set on the
    /// first `device("mlx") { }` entry under that condition; the
    /// warning is emitted at most once per `Environment`.
    pub mlx_fallback_warned: bool,
    /// User-visible notices collected during one eval (e.g. "adam under
    /// device(\"cuda\") ran on CPU"). Drained at the eval boundary and
    /// prepended to the result so the user sees when a GPU device
    /// silently fell back to CPU. Deduped within an eval.
    pub notices: Vec<String>,
    /// Optional remote peer dispatcher installed by the orchestrator
    /// around one eval call. Local eval leaves this unset.
    pub peer_dispatcher: Option<Arc<dyn PeerDispatcher>>,
    /// The GPU optimizer step for this build, installed by `new`.
    /// `eval_adam` calls it device-agnostically. Only exists on a GPU
    /// build (CUDA on Linux, MLX on Apple); a CPU-only build has no
    /// `gpu_step` module/field at all.
    #[cfg(any(
        all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
        all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
    ))]
    pub gpu_step: Option<Arc<dyn mlpl_eval_state::GpuAdamStep>>,
    /// Peer-resident tensor handles bound by assignment.
    pub device_tensors: HashMap<String, Value>,
    /// Saga typed-packed-bytes: `Value::Bytes` packed buffers bound by
    /// assignment. Sibling to `device_tensors` -- a full-`Value` store
    /// keyed by name, distinct from the f64 array/string tables.
    pub bytes: HashMap<String, Value>,
    /// Native-extension handles bound by assignment. Sibling to
    /// `device_tensors` / `bytes` -- a full-`Value` store keyed by
    /// name, holding a `Value::ExtHandle` a provider returned.
    pub ext_handles: HashMap<String, Value>,
    /// Native-port channel endpoints keyed by handle slot (Saga
    /// extensions-event-loop). Share-nothing: only owned `Value`s cross
    /// the channels, so the interpreter and a far-end thread cannot
    /// race. The far end is a provider UI thread or a test echo worker.
    pub ports: HashMap<u32, PortEndpoints>,
    /// Next port slot to mint (monotonic per interpreter).
    pub next_port_slot: u32,
    /// Saga 23 step 001: optional `ValueTag` attached per binding
    /// name. Auto-tagged by producer ops in steps 002+; consumed
    /// by predicate-checked consumers, `:describe` / `:vars` /
    /// `:tags`, and trace JSON. Re-binding a name with a new tag
    /// overwrites; clearing is explicit via `clear_tag`. Untyped
    /// bindings keep working unchanged (gradual-typing
    /// additivity rule).
    pub tags: HashMap<String, ValueTag>,
    /// Bindings that hold a `BuiltinRef` value (`:add`, `:max`,
    /// `:+`, etc.). Sibling to `vars` / `strings` -- a separate
    /// namespace so user variables can't shadow builtin
    /// references and vice versa. `f = :add` lands here;
    /// `eval_expr(Expr::Ident(f))` looks up here after `vars`
    /// and `strings`. Forward-compatible with first-class
    /// functions: when `Value::Function` lands, this map either
    /// gets absorbed or stays as the "named-builtin" subset.
    pub builtin_refs: HashMap<String, String>,
    /// Saga 21.5 step 001: optional live-metric sink invoked by
    /// `eval_train` after each iteration with every binding ending
    /// in `_metric` whose value is a scalar. Local REPL evals leave
    /// this `None`; the SSE `/eval_stream` handler installs a
    /// channel-backed sink so the events stream to the client.
    pub metric_sink: Option<Arc<dyn MetricSink>>,
    /// Saga 21.5 step 003: optional cooperative-cancellation token
    /// checked at loop heads (`for`, `train`, `repeat`) and before
    /// every builtin dispatch. Local REPL evals leave this `None`;
    /// the SSE `/eval_stream` and `/eval` handlers install one per
    /// call so the shared `/cancel` endpoint can trip the bool
    /// from a different thread.
    pub interrupt: Option<Interrupt>,
    /// Saga 29 step 001: record-valued bindings. Sibling to `vars`
    /// / `strings` / `models` -- a separate namespace so the
    /// existing one-Value-variant-per-map pattern holds.
    /// `r = {X: 1, Y: 2}` lands here; `eval_expr(Expr::Ident("r"))`
    /// looks up here before falling through to other variant maps.
    pub records: HashMap<String, BTreeMap<String, Value>>,
    /// Saga 29 step 002: string-list-valued bindings. Same
    /// namespace pattern as `records`. `names = ["cat", "dog"]`
    /// lands here.
    pub string_lists: HashMap<String, Vec<String>>,
    pub results: HashMap<String, (bool, Value)>, // Saga 29 step 012
    /// Source-ordered `@test` registry (reflection: `tests()` /
    /// `test_info`). Registration happens at def evaluation.
    pub tests: Vec<mlpl_eval_state::TestEntry>,
    /// Display name of the source currently evaluating (script
    /// chunks set this; None renders as "repl").
    pub current_source: Option<String>,
    pub cli_args: Vec<String>, // Saga 31 step 003
    pub user_fns: HashMap<String, mlpl_eval_state::UserFn>,
    /// Raw text of the program currently being evaluated, when the
    /// entry point provided it (naming-and-docs saga): `def` slices
    /// its own span out of this so `:list` shows the source AS
    /// WRITTEN, `#` comments and all -- the APL2 listing experience.
    pub pending_source: Option<String>,
}

impl Environment {
    /// Create an empty environment. On a GPU build it installs this
    /// build's GPU optimizer step so `eval_adam` can dispatch to it.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(any(
            all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
            all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
        ))]
        {
            Self {
                gpu_step: mlpl_eval_state::installed_gpu_step(),
                ..Self::default()
            }
        }
        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
            all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
        )))]
        {
            Self::default()
        }
    }

    /// Register a port's channel endpoints and return its opaque handle
    /// (a `Value::ExtHandle`). Mint-on-open: MLPL has no constructor for
    /// a port, so one cannot be forged from numbers. The far end holds
    /// the matching `Receiver` / `Sender`.
    pub fn register_port(&mut self, commands: Sender<Value>, events: Receiver<Value>) -> Value {
        let slot = self.next_port_slot;
        self.next_port_slot = slot.wrapping_add(1);
        self.ports.insert(slot, PortEndpoints { commands, events });
        Value::ExtHandle {
            extension_id: PORT_EXTENSION_ID,
            type_id: PORT_TYPE_ID,
            slot,
            generation: 1,
        }
    }

    /// Resolve a port handle to its live endpoints.
    ///
    /// # Errors
    /// Returns `ExtensionError` for a non-port value or an
    /// unregistered / closed slot.
    pub fn resolve_port(
        &self,
        handle: &Value,
    ) -> Result<&PortEndpoints, mlpl_eval_types::EvalError> {
        let err = |message: String| mlpl_eval_types::EvalError::ExtensionError {
            function: "port".into(),
            message,
        };
        match handle {
            Value::ExtHandle {
                extension_id, slot, ..
            } if *extension_id == PORT_EXTENSION_ID => self
                .ports
                .get(slot)
                .ok_or_else(|| err("unknown or closed port".into())),
            other => Err(err(format!(
                "expected a port handle, got {}",
                mlpl_eval_types::value_kind(other)
            ))),
        }
    }

    /// This build's GPU optimizer step, if any (cloned `Arc`, so the
    /// caller can hold it while mutably borrowing the env for the step).
    #[cfg(any(
        all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
        all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
    ))]
    #[must_use]
    pub fn gpu_step(&self) -> Option<Arc<dyn mlpl_eval_state::GpuAdamStep>> {
        self.gpu_step.clone()
    }
}

/// Public test helper: walk the parameter tree of the model bound to
/// `name` and return the flat list of param identifiers it owns.
#[must_use]
pub fn model_params(env: &Environment, name: &str) -> Option<Vec<String>> {
    env.models.get(name).map(ModelSpec::params)
}

/// Options for the `run_script` builtin, resolved by the
/// evaluator (paths already sandbox-checked and absolute).
#[derive(Clone, Debug, Default)]
pub struct RunScriptOpts {
    /// Include/sandbox root for the child script.
    pub source_dir: Option<std::path::PathBuf>,
    /// Dataset directory for the child script.
    pub data_dir: Option<std::path::PathBuf>,
    /// Buffer typed test events and return them in the result.
    pub capture: bool,
}

/// The host-side implementation `run_script` dispatches to.
pub type RunScriptHook = fn(&std::path::Path, &RunScriptOpts) -> Value;
