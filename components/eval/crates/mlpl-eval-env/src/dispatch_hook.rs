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
/// `HasDispatch` -- or a cluster crate -- without ever entering the
/// evaluator).
///
/// # Errors
/// The hub's dispatch errors, or the not-installed sentinel.
pub fn dispatch_or_err(
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

/// The hub's expression evaluator: cluster crates (models, ...) call
/// back into `eval_expr` through this hook, mirroring the dispatch
/// inversion above.
pub type EvalFn = fn(
    &mlpl_parser::Expr,
    &mut Environment,
    &mut Option<&mut mlpl_trace::Trace>,
) -> Result<mlpl_eval_types::Value, EvalError>;

static EVAL: OnceLock<EvalFn> = OnceLock::new();

/// Install the process's expression evaluator. Idempotent.
pub fn install_eval(f: EvalFn) {
    let _ = EVAL.set(f);
}

/// Route through the installed evaluator; a missing install is a
/// loud error naming the seam.
///
/// # Errors
/// The hub's eval errors, or the not-installed sentinel.
pub fn eval_or_err(
    expr: &mlpl_parser::Expr,
    env: &mut Environment,
    trace: &mut Option<&mut mlpl_trace::Trace>,
) -> Result<mlpl_eval_types::Value, EvalError> {
    match EVAL.get() {
        Some(f) => f(expr, env, trace),
        None => Err(EvalError::Unsupported(
            "eval hook not installed (mlpl_eval::eval_program installs it)".into(),
        )),
    }
}
