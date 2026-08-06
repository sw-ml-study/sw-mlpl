//! Per-session runtime state leaves: the cooperative cancellation
//! token, optimizer moment buffers, and experiment records. Pure
//! data -- the evaluator-coupled logic (loop checkpoints, the
//! `experiment` block evaluator, optimizer steps) stays above.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mlpl_array::DenseArray;
use mlpl_tensor_handle::TensorHandle;
use serde::{Deserialize, Serialize};

/// Cancellation token (Saga 21.5 step 003). Cheap to `clone`
/// (shared `Arc`). The same instance is held by the session map
/// (cancel handler flips it) and the evaluator's `Environment`
/// (eval reads it at loop heads and before builtin dispatch).
#[derive(Debug, Clone, Default)]
pub struct Interrupt(Arc<AtomicBool>);

impl Interrupt {
    /// Construct a fresh, not-yet-tripped token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Trip the token. Idempotent: a second `set()` is a no-op,
    /// which is what makes `POST /cancel` idempotent at the server
    /// layer too.
    pub fn set(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Clear the token back to "not tripped". Called by the server
    /// before each eval so a prior cancel does not contaminate the
    /// next call on the same session.
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }

    /// Read the current state. Used by `Environment::check_interrupt`
    /// at each loop / pre-builtin checkpoint.
    #[must_use]
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Per-optimizer, per-parameter state buffers (e.g. momentum
/// velocity, Adam first/second moments). Storage is plain public
/// fields keyed by `(optimizer_name, param_name, slot_name)` so the
/// optimizer steps can fill in `momentum_sgd` and `adam` without
/// dragging accessor helpers across function-count budgets.
#[derive(Clone, Debug, Default)]
pub struct OptimizerState {
    /// Buffers keyed by `(optimizer_name, param_name, slot_name)`.
    /// `slot_name` lets a single optimizer store multiple buffers per
    /// param (e.g. Adam needs both `m` and `v`).
    pub buffers: HashMap<(String, String, String), DenseArray>,
    /// Per-optimizer step counter (for Adam bias correction).
    pub steps: HashMap<String, u64>,
    /// Device-resident optimizer state (saga E4 step 006): moments
    /// and the weight cache stay on the backend across the whole
    /// training loop, keyed like `buffers` (weights use slot "w").
    /// The resident path OWNS a slot once it writes it; the host
    /// `buffers` entries it supersedes are removed.
    pub resident: HashMap<(String, String, String), TensorHandle>,
    /// Cache witness for the "w" slots: the data pointer of the
    /// host mirror (`env` vars) the resident weight was synced
    /// with. Any foreign write to the var changes the pointer and
    /// invalidates the cached handle (re-upload on next step).
    pub resident_witness: HashMap<String, usize>,
}

/// One recorded run. Written to `<exp_dir>/<name>/<ts>/run.json`
/// by the terminal REPL; also appended to `env.experiment_log`
/// so the web REPL can surface runs via `:experiments`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentRecord {
    /// Name passed to the `experiment "..."` form.
    pub name: String,
    /// Wall-clock `SystemTime::duration_since(UNIX_EPOCH)` in
    /// nanoseconds at run-entry. Used to make the on-disk
    /// timestamp subdir unique.
    pub timestamp_ns: u128,
    /// `_metric`-suffixed scalar values captured at run-exit.
    pub metrics: BTreeMap<String, f64>,
    /// Shape metadata for every bound tracked parameter at
    /// run-exit. Keyed by param name.
    pub params_snapshot: BTreeMap<String, ParamShape>,
}

/// Shape snapshot stored inside an `ExperimentRecord`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamShape {
    /// Positional dims.
    pub shape: Vec<usize>,
    /// Per-axis labels when the array is labeled; `None` when
    /// the array has no labels.
    pub labels: Option<Vec<Option<String>>>,
}
