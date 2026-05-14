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
/// `reqwest::blocking::Response`; the WASM impl wraps a
/// `Cursor<String>` since gloo's fetch buffers the body before
/// exposing it.
pub fn parse_sse_stream<R: std::io::BufRead>(reader: R, on_metric: &mut MetricCb) -> StreamOutcome {
    let mut event: Option<String> = None;
    let mut data: Option<String> = None;
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                return StreamOutcome::Error {
                    message: format!("stream read: {e}"),
                };
            }
        };
        if line.is_empty() {
            if let Some(outcome) = dispatch_sse_frame(event.take(), data.take(), on_metric) {
                return outcome;
            }
        } else if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data = Some(rest.trim().to_string());
        }
    }
    StreamOutcome::Error {
        message: "stream ended without terminal frame".into(),
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
    match event.as_str() {
        "ready" => None,
        "metric" => {
            on_metric(&RemoteMetric {
                name: v.get("name").and_then(|x| x.as_str())?.to_string(),
                step: v.get("step")?.as_u64()? as usize,
                value: v.get("value")?.as_f64()?,
            });
            None
        }
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
