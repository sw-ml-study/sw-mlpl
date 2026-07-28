//! Saga 33 step 003: metric-sink install / clear / borrow +
//! `emit_metrics` extracted from `env.rs`. Saga 21.5 step 001
//! added the optional `MetricSink` so the SSE `/eval_stream`
//! handler can push every `_metric`-suffixed scalar to clients;
//! local REPL evals leave it `None`.

use std::sync::Arc;

use mlpl_eval_core::metric_sink::MetricSink;

use crate::env::Environment;

impl Environment {
    /// Saga 21.5 step 001: install a live `MetricSink`. Called by
    /// the SSE `/eval_stream` handler around one eval call;
    /// `clear_metric_sink` removes it on exit.
    pub fn set_metric_sink(&mut self, sink: Arc<dyn MetricSink>) {
        self.metric_sink = Some(sink);
    }

    /// Saga 21.5 step 001: remove any installed `MetricSink`.
    pub fn clear_metric_sink(&mut self) {
        self.metric_sink = None;
    }

    /// Saga 21.5 step 001: borrow the installed `MetricSink`, if
    /// any. `eval_train` clones the `Arc` per iteration so the
    /// emission loop can call out without holding an immutable
    /// borrow of the environment alongside the var iterator.
    #[must_use]
    pub fn metric_sink(&self) -> Option<Arc<dyn MetricSink>> {
        self.metric_sink.clone()
    }

    /// Saga 21.5 step 001: emit every `_metric`-suffixed scalar
    /// binding for iteration `step` through the installed
    /// `MetricSink`. Called by `eval_train` at the end of each
    /// iteration; no-op when no sink is installed. Extracted
    /// from `eval_train` to keep that function under the
    /// sw-checklist 50-line budget.
    ///
    /// Connect-telemetry step 003: when the block defines NO
    /// explicit `*_metric` binding, the train loop's own per-step
    /// loss (`step_loss` -- already computed, no recompute) is
    /// emitted as the implicit `loss` metric, so every streamed
    /// train feeds the live loss panel. Explicit bindings suppress
    /// the implicit frame.
    pub fn emit_metrics(&self, step: usize, step_loss: f64) {
        let Some(sink) = self.metric_sink() else {
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
