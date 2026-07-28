//! The one upward seam from the env layer to the evaluator: device
//! dispatch. `HasDispatch for Environment` (`env_trait_impls_dispatch`)
//! must live in this crate with the type, but the dispatch decision
//! tree (`device_dispatch::dispatched_call`) is interpreter-coupled
//! and stays in `mlpl-eval` -- so the hub INSTALLS its function here
//! at eval entry (the same inversion `gpu_step`'s registry uses).

use std::sync::OnceLock;

use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;

use crate::env::Environment;

/// The hub's dispatch entry: run a named builtin with `env`'s active
/// device in mind.
pub type DispatchFn = fn(&Environment, &str, Vec<DenseArray>) -> Result<DenseArray, EvalError>;

static DISPATCH: OnceLock<DispatchFn> = OnceLock::new();

/// Install the process's dispatch function. Idempotent: first write
/// wins, so every eval entry point can call it unconditionally.
pub fn install_dispatch(f: DispatchFn) {
    let _ = DISPATCH.set(f);
}

/// Route through the installed dispatch function; a missing install
/// is a loud error naming the seam (only reachable if a caller uses
/// `HasDispatch` without ever entering the evaluator).
pub(crate) fn dispatch_or_err(
    env: &Environment,
    op: &str,
    args: Vec<DenseArray>,
) -> Result<DenseArray, EvalError> {
    match DISPATCH.get() {
        Some(f) => f(env, op, args),
        None => Err(EvalError::Unsupported(
            "device dispatch hook not installed (mlpl_eval::eval_program installs it)".into(),
        )),
    }
}
