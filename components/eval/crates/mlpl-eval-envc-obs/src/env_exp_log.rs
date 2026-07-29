//! Saga 33 step 003: experiment-log methods extracted from
//! `env.rs`. The terminal REPL appends a record each time the
//! `experiment` builtin completes; the web REPL keeps the log
//! in-memory for the inspect popup.

use mlpl_eval_env::Environment;
use mlpl_eval_state::ExperimentRecord;

impl EnvExpLog for Environment {
    fn push_experiment_log(&mut self, rec: ExperimentRecord) {
        self.experiment_log.push(rec);
    }

    fn experiment_log(&self) -> &[ExperimentRecord] {
        &self.experiment_log
    }
}

/// The in-memory experiment log.
pub trait EnvExpLog {
    /// Append a completed experiment record to the log.
    fn push_experiment_log(&mut self, rec: ExperimentRecord);
    /// Borrow every recorded experiment in order.
    fn experiment_log(&self) -> &[ExperimentRecord];
}

impl EnvMetricEmit for Environment {
    fn emit_metrics(&self, step: usize, step_loss: f64) {
        let Some(sink) = self.metric_sink.as_ref() else {
            return;
        };
        let metrics: Vec<(String, f64)> = self
            .vars
            .iter()
            .filter(|(name, arr)| name.ends_with("_metric") && arr.rank() == 0)
            .map(|(name, arr)| (name.clone(), arr.data()[0]))
            .collect();
        if metrics.is_empty() {
            sink.emit("loss", step, step_loss);
            return;
        }
        for (name, value) in metrics {
            sink.emit(&name, step, value);
        }
    }
}

/// Emit every `_metric`-suffixed scalar (or the step loss) through
/// the installed sink; lives with the experiment log because both
/// are the observation surface of a training run.
pub trait EnvMetricEmit {
    fn emit_metrics(&self, step: usize, step_loss: f64);
}
