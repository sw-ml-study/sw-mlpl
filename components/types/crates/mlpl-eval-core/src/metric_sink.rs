//! Live metric capture during `train { ... }` (Saga 21.5 step 001).
//!
//! `eval_train` calls `sink.emit(name, step, value)` for every binding
//! ending in `_metric` after each loop iteration. The default
//! `Environment` has no sink installed and the call is skipped, so
//! existing callers see no behavior change.
//!
//! `mlpl-serve`'s SSE `/eval_stream` handler installs a sink whose
//! `emit` pushes onto a `tokio::sync::mpsc::Sender`, turning the
//! per-iteration scalar capture into a streamed `event: metric` frame.

/// Receiver for per-iteration `_metric` scalar emissions inside a
/// `train { ... }` block.
pub trait MetricSink: Send + Sync + std::fmt::Debug {
    /// Called once per `_metric`-suffixed scalar binding after each
    /// `train { }` iteration. `name` is the binding name verbatim
    /// (`"loss_metric"`, `"acc_metric"`); `step` is the zero-based
    /// iteration index; `value` is the scalar contents of the
    /// `DenseArray`.
    fn emit(&self, name: &str, step: usize, value: f64);
}
