//! Saga 33 step 003: experiment-log methods extracted from
//! `env.rs`. The terminal REPL appends a record each time the
//! `experiment` builtin completes; the web REPL keeps the log
//! in-memory for the inspect popup.

use crate::env::Environment;
use crate::experiment::ExperimentRecord;

impl Environment {
    /// Append a completed experiment record to the log.
    pub fn push_experiment_log(&mut self, rec: ExperimentRecord) {
        self.experiment_log.push(rec);
    }

    /// Borrow every recorded experiment in order.
    #[must_use]
    pub fn experiment_log(&self) -> &[ExperimentRecord] {
        &self.experiment_log
    }
}
