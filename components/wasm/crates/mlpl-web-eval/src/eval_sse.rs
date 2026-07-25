//! Saga 21.5 step 008: shared SSE wire-frame parser.
//!
//! `parse_sse_stream` reads the response body line-by-line and
//! dispatches assembled `event:`/`data:` frames through
//! `dispatch_sse_frame` to either the `on_metric` callback (for
//! `event: metric`), a terminal `StreamOutcome::Done` (for
//! `event: done`), `Cancelled` (for `event: cancelled`), or
//! `Error` (for `event: error` or malformed payloads).
//!
//! Lives in its own module so the native + WASM impls of
//! `RemoteEvaluator::eval_stream` can share one implementation
//! without inflating `eval.rs` over its 500-line file budget.

use crate::eval::{MetricCb, RemoteMetric, StreamOutcome};

/// Consume an SSE response body via a `BufRead` line iterator
/// and return the terminal outcome once a `done` / `cancelled`
/// / `error` frame arrives. The native impl wraps a
/// `reqwest::blocking::Response`; the WASM streaming impl feeds
/// `ReadableStream` chunks through [`SseFeed`] directly.
pub fn parse_sse_stream<R: std::io::BufRead>(reader: R, on_metric: &mut MetricCb) -> StreamOutcome {
    let mut feed = SseFeed::default();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                return StreamOutcome::Error {
                    message: format!("stream read: {e}"),
                };
            }
        };
        if let Some(outcome) = feed.push(format!("{line}\n").as_bytes(), on_metric) {
            return outcome;
        }
    }
    StreamOutcome::Error {
        message: "stream ended without terminal frame".into(),
    }
}

/// Push-based twin of [`parse_sse_stream`] for the browser
/// `ReadableStream` path: chunks arrive at arbitrary byte boundaries
/// (mid-line, mid-frame, mid-UTF-8), get buffered until a full line is
/// available, and completed frames dispatch exactly like the pull
/// parser's. Connect-telemetry step 002.
#[derive(Default)]
pub struct SseFeed {
    buf: Vec<u8>,
    event: Option<String>,
    data: Option<String>,
}

impl SseFeed {
    /// Feed one chunk of SSE body bytes. Fires `on_metric` per
    /// completed `event: metric` frame; returns the terminal outcome
    /// once a `done` / `cancelled` / `error` frame completes.
    pub fn push(&mut self, chunk: &[u8], on_metric: &mut MetricCb) -> Option<StreamOutcome> {
        self.buf.extend_from_slice(chunk);
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&raw);
            if let Some(outcome) = self.take_line(line.trim_end_matches(['\n', '\r']), on_metric) {
                return Some(outcome);
            }
        }
        None
    }

    /// One complete line: blank dispatches the assembled frame,
    /// `event:` / `data:` prefixes accumulate into it.
    fn take_line(&mut self, line: &str, on_metric: &mut MetricCb) -> Option<StreamOutcome> {
        if line.is_empty() {
            return dispatch_sse_frame(self.event.take(), self.data.take(), on_metric);
        }
        if let Some(rest) = line.strip_prefix("event:") {
            self.event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            self.data = Some(rest.trim().to_string());
        }
        None
    }
}

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

fn dispatch_sse_frame(
    event: Option<String>,
    data: Option<String>,
    on_metric: &mut MetricCb,
) -> Option<StreamOutcome> {
    let (Some(event), Some(data)) = (event, data) else {
        return None;
    };
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;
    if event == "metric" {
        on_metric(&RemoteMetric {
            name: v.get("name").and_then(|x| x.as_str())?.to_string(),
            step: v.get("step")?.as_u64()? as usize,
            value: v.get("value")?.as_f64()?,
        });
        return None;
    }
    terminal_outcome(&event, &v)
}

/// A `done` / `cancelled` / `error` frame's terminal outcome (`None`
/// for `ready` and unknown events).
fn terminal_outcome(event: &str, v: &serde_json::Value) -> Option<StreamOutcome> {
    match event {
        "done" => Some(StreamOutcome::Done {
            value: v.get("value").and_then(|x| x.as_str())?.to_string(),
            kind: v.get("kind").and_then(|x| x.as_str())?.to_string(),
        }),
        "cancelled" => Some(StreamOutcome::Cancelled {
            step: v.get("step")?.as_u64()? as usize,
            partial_losses: v
                .get("partial_losses")?
                .as_array()?
                .iter()
                .filter_map(serde_json::Value::as_f64)
                .collect(),
        }),
        "error" => Some(StreamOutcome::Error {
            message: v
                .get("error")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown error")
                .to_string(),
        }),
        _ => None,
    }
}
