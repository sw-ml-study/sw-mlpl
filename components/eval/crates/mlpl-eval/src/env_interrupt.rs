//! Saga 33 step 003: cooperative-cancellation token methods
//! extracted from `env.rs`. Saga 21.5 step 003 added the
//! optional `Interrupt`; the SSE `/eval_stream` and `/eval`
//! handlers install one per call so the shared `/cancel`
//! endpoint can trip the bool from a different thread.

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use crate::interrupt::Interrupt;

impl Environment {
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
}
