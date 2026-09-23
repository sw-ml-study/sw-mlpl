//! Optimizer state: the per-(optimizer, param, slot) buffer map lives on
//! `Environment` (Saga 10 design choice -- Adam / momentum-SGD are thin
//! wrappers around `grad` plus per-param buffers, so no separate crate).

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use mlpl_eval_types::EvalError;

// The buffer type moved to mlpl-eval-state (env-types-out step);
// re-exported so `crate::grad::OptimizerState` paths keep working.
pub use mlpl_eval_state::OptimizerState;

/// `reset_optimizer()` -- drop all optimizer moment buffers and step counters
/// so a script can train another variant from a clean slate in one process
/// (moe-microscope F22). Returns 0.
pub(crate) fn eval_reset_optimizer(
    args: &[Expr],
    env: &mut Environment,
) -> Result<mlpl_eval_types::Value, EvalError> {
    if !args.is_empty() {
        return Err(EvalError::BadArity {
            func: "reset_optimizer".into(),
            expected: 0,
            got: args.len(),
        });
    }
    env.optim_state.clear();
    Ok(mlpl_eval_types::Value::Array(DenseArray::from_scalar(0.0)))
}

/// Read-only accessor used by tests and downstream optimizer code.
#[must_use]
pub fn optim_state(env: &Environment) -> &OptimizerState {
    &env.optim_state
}

/// Mutable accessor used by tests and downstream optimizer code.
pub fn optim_state_mut(env: &mut Environment) -> &mut OptimizerState {
    &mut env.optim_state
}
