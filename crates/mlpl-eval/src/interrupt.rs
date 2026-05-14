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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mlpl_array::{DenseArray, Shape};
use mlpl_trace::TraceValue;

use crate::env::Environment;
use crate::error::EvalError;

/// Cancellation token. Cheap to `clone` (shared `Arc`). The same
/// instance is held by the session map (cancel handler flips it)
/// and the evaluator's `Environment` (eval reads it).
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
