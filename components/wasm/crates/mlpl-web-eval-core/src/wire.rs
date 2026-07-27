//! Eval wire types shared across the web eval stack (spike step
//! 015): result/metric/stream callbacks and the terminal stream
//! outcome, plus the fetch-deadline helper the wasm fetchers race
//! their requests against.

/// Callback fired with the formatted result of one eval. Matches
/// the existing `WasmSession::eval(&str) -> String` shape: errors
/// arrive as `"error: <msg>"` strings, not as a separate Err
/// variant, so call sites can keep their existing
/// `result.starts_with("error:")` test for the red-text UI.
pub type ResultCb = Box<dyn FnOnce(String) + 'static>;

/// Saga 21.5 step 007: per-iteration metric frame surfaced by
/// `RemoteEvaluator::eval_stream`. Mirrors the server-side
/// `SseEvent::Metric` payload.
#[derive(Debug, Clone)]
pub struct RemoteMetric {
    pub name: String,
    pub step: usize,
    pub value: f64,
}

/// Saga 21.5 step 007: terminal SSE frame. `Done` carries the
/// final value + kind (matching the non-streaming `/eval`);
/// `Cancelled` carries the step + partial loss curve from a
/// `train { }` that observed a cancel; `Error` covers both
/// HTTP-level failures (auth, lex/parse) and runtime
/// `EvalError`s emitted as `event: error`.
#[derive(Debug)]
pub enum StreamOutcome {
    Done {
        value: String,
        kind: String,
    },
    Cancelled {
        step: usize,
        partial_losses: Vec<f64>,
    },
    Error {
        message: String,
    },
}

/// Callback fired once per `event: metric` SSE frame during
/// streaming eval.
pub type MetricCb = Box<dyn FnMut(&RemoteMetric) + 'static>;

/// Callback fired exactly once with the terminal outcome of a
/// streaming eval.
pub type StreamCb = Box<dyn FnOnce(StreamOutcome) + 'static>;

#[cfg(target_arch = "wasm32")]
/// Race a connect-server request against a timeout so a wedged or absent
/// `mlpl-serve` fails FAST with a clear message instead of hanging the
/// REPL (the connect-only `:ask` / CUDA / MLX demos all go through here).
pub async fn with_deadline<T>(
    ms: u32,
    what: &str,
    fut: impl std::future::Future<Output = Result<T, String>>,
) -> Result<T, String> {
    let work = Box::pin(fut);
    let timer = Box::pin(gloo::timers::future::TimeoutFuture::new(ms));
    match futures::future::select(work, timer).await {
        futures::future::Either::Left((res, _)) => res,
        futures::future::Either::Right(((), _)) => Err(format!(
            "{what}: connect server did not respond within {}s -- is it up? \
             (check the ?connect= URL, or restart mlpl-serve)",
            ms / 1000
        )),
    }
}

// Moved with the type (spike step 015): terminal-outcome display.
impl StreamOutcome {
    /// Collapse a terminal stream outcome into the REPL's
    /// `(display, is_error)` convention (errors as `"error: ..."`
    /// text), so streaming call sites plug into the same history
    /// entries the non-streaming path produces.
    #[must_use]
    pub fn into_display(self) -> (String, bool) {
        match self {
            Self::Done { value, .. } => (value, false),
            Self::Cancelled {
                step,
                partial_losses,
            } => (
                format!(
                    "cancelled at step {step} ({} partial loss points kept)",
                    partial_losses.len()
                ),
                false,
            ),
            Self::Error { message } => (format!("error: {message}"), true),
        }
    }
}
