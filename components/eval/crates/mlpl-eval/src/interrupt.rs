//! Cooperative cancellation token (Saga 21.5 step 003).
//!
//! `mlpl-serve`'s `/cancel` handler flips a session-scoped
//! `AtomicBool` shared with the in-flight evaluator. The
//! evaluator checks the bool at the head of every loop iteration
//! (`for`, `train`, `repeat`) plus before every builtin dispatch
//! and raises `EvalError::Cancelled` on trip. Builtins that run
//! over large arrays do not yield mid-call, so the latency floor
//! is "one op" -- a giant `matmul` finishes before the next
//! pre-dispatch check observes the cancel. Documented in the
//! contract.
//!
//! The default `Environment` has no `Interrupt` installed; the
//! check is a no-op and existing callers see no behavior change
//! (gradual-additivity rule). The server installs one for the
//! duration of each eval call.

use crate::env_api::*;
use mlpl_array::{DenseArray, Shape};
use mlpl_trace::TraceValue;

use crate::env::Environment;
use mlpl_eval_types::EvalError;

// The token type moved to mlpl-eval-state (env-types-out step);
// re-exported so `crate::interrupt::Interrupt` paths keep working.
pub use mlpl_eval_state::Interrupt;

/// Enrich a bubbled `Cancelled` (from a pre-builtin checkpoint
/// inside a `train` body or from the loop-head check itself)
/// with the current iteration index and the accumulated loss
/// curve, and persist `last_losses` on the env so post-cancel
/// `:vars` still surfaces the partial vector. Returns the
/// enriched error in `eval_train`'s `Result` shape so the call
/// site stays a single `return` expression.
pub(crate) fn enrich_train_cancel(
    env: &mut Environment,
    step: usize,
    losses: Vec<f64>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let losses_arr = DenseArray::new(Shape::new(vec![losses.len()]), losses.clone())
        .expect("losses shape matches data");
    env.set("last_losses".into(), losses_arr);
    Err(EvalError::Cancelled {
        step,
        partial_losses: losses,
    })
}
