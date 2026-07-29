//! Saga 33 step 003: metric-sink install / clear / borrow +
//! `emit_metrics` extracted from `env.rs`. Saga 21.5 step 001
//! added the optional `MetricSink` so the SSE `/eval_stream`
//! handler can push every `_metric`-suffixed scalar to clients;
//! local REPL evals leave it `None`.

use std::sync::Arc;

use mlpl_eval_core::metric_sink::MetricSink;

use mlpl_eval_env::Environment;

impl EnvMetricSink for Environment {
    fn set_metric_sink(&mut self, sink: Arc<dyn MetricSink>) {
        self.metric_sink = Some(sink);
    }

    fn clear_metric_sink(&mut self) {
        self.metric_sink = None;
    }

    fn metric_sink(&self) -> Option<Arc<dyn MetricSink>> {
        self.metric_sink.clone()
    }
}

/// Metric-sink installation (live loss/metric streaming).
pub trait EnvMetricSink {
    /// Saga 21.5 step 001: install a live `MetricSink`. Called by
    /// the SSE `/eval_stream` handler around one eval call;
    /// `clear_metric_sink` removes it on exit.
    fn set_metric_sink(&mut self, sink: Arc<dyn MetricSink>);
    /// Saga 21.5 step 001: remove any installed `MetricSink`.
    fn clear_metric_sink(&mut self);
    /// Saga 21.5 step 001: borrow the installed `MetricSink`, if
    /// any. `eval_train` clones the `Arc` per iteration so the
    /// emission loop can call out without holding an immutable
    /// borrow of the environment alongside the var iterator.
    fn metric_sink(&self) -> Option<Arc<dyn MetricSink>>;
}
