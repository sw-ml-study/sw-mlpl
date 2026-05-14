//! Saga 21.5 step 002: streaming `--connect` mode.
//!
//! Two responsibilities live here so the crate stays at
//! its sw-checklist module budget:
//!
//! 1. Connect-mode argv + env-var parsing
//!    (`parse_connect_args`). Pure function so the
//!    integration tests can drive it without a
//!    subprocess.
//! 2. The SSE wire path (`eval_remote_stream`) plus the
//!    per-iteration progress display (`render_metric`).
//!    Pure transport -- the REPL loop that calls into
//!    these lives in `connect_repl.rs`.
//!
//! Why pure-fn parsing: the binary's wrapper translates
//! `Err` into `std::process::exit(2)`, but the test
//! suite needs to assert the exit reason without forking
//! a subprocess.

use std::io::{self, BufRead, BufReader, Write};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::connect::{ClientError, EvalResponse};

/// Window for the double-Ctrl-C cancel debounce. The second press
/// within this window triggers a `/cancel` POST; presses outside
/// the window reset the counter to "first press".
pub const CANCEL_DOUBLE_WINDOW: Duration = Duration::from_secs(2);

/// Three connect-mode invocation shapes the binary
/// distinguishes after argv + env-var parsing. `Local`
/// means "no `--connect` flag, fall through to local
/// REPL"; `Remote` is connect mode with the streaming
/// flag baked in.
#[derive(Debug)]
pub enum ConnectMode {
    Local,
    Remote { url: String, stream: bool },
}

/// Parser errors that the binary maps to `exit(2)` with
/// a usage hint. Kept enum-shaped (rather than `String`)
/// so tests can match on variants without coupling to
/// the exact wording of the message.
#[derive(Debug)]
pub enum ConnectArgsError {
    StreamWithoutConnect,
    LocalFlagWithConnect(String),
}

/// One `event: metric` frame off the SSE stream. The
/// streaming client surfaces these to its caller via a
/// `FnMut(&SseMetric)` callback so the REPL can render
/// one progress line per iteration while the eval is
/// still running on the server.
#[derive(Debug, Clone)]
pub struct SseMetric {
    pub name: String,
    pub step: usize,
    pub value: f64,
}

const LOCAL_ONLY_FLAGS: &[&str] = &["-f", "--file", "--data-dir", "--exp-dir"];

/// Parse the connect-mode shape from argv plus the
/// `MLPL_REPL_STREAM` env var. Pure function: the binary
/// reads the env var itself and passes it in; tests can
/// vary it freely. Falsy values (`0`, empty, `false`,
/// `no`, `off`, case-insensitive) explicitly disable
/// streaming so a stale `export` in a user's shell does
/// not silently change behavior.
pub fn parse_connect_args(
    args: &[String],
    stream_env: Option<&str>,
) -> Result<ConnectMode, ConnectArgsError> {
    let connect_url = args
        .iter()
        .position(|a| a == "--connect")
        .and_then(|p| args.get(p + 1))
        .cloned();
    let env_truthy = stream_env.is_some_and(|s| {
        let lower = s.trim().to_ascii_lowercase();
        !lower.is_empty() && lower != "0" && lower != "false" && lower != "no" && lower != "off"
    });
    let stream = args.iter().any(|a| a == "--stream") || env_truthy;
    match connect_url {
        None if stream => Err(ConnectArgsError::StreamWithoutConnect),
        None => Ok(ConnectMode::Local),
        Some(url) => {
            if let Some(bad) = args.iter().find(|a| LOCAL_ONLY_FLAGS.contains(&a.as_str())) {
                return Err(ConnectArgsError::LocalFlagWithConnect(bad.clone()));
            }
            Ok(ConnectMode::Remote { url, stream })
        }
    }
}

/// Render one `metric` frame as a progress line. Uses
/// `\r` (no newline) so successive frames overwrite the
/// same terminal row -- the REPL wrapper writes a
/// trailing `\n` after the eval completes so the result
/// lands on its own line. Test callers compare on
/// substrings.
pub fn render_metric<W: Write>(out: &mut W, m: &SseMetric) -> io::Result<()> {
    write!(out, "  [step {}] {} = {:.6}\r", m.step, m.name, m.value)?;
    out.flush()
}

#[derive(Deserialize)]
struct MetricData {
    name: String,
    step: usize,
    value: f64,
}

#[derive(Deserialize)]
struct DoneData {
    value: String,
    kind: String,
}

#[derive(Deserialize)]
struct ErrorData {
    error: String,
}

#[derive(Deserialize)]
struct CancelledData {
    step: usize,
    partial_losses: Vec<f64>,
}

/// Saga 21.5 step 003: debounce predicate for the
/// double-Ctrl-C cancel binding. Returns `true` when the current
/// press should trigger a cancel POST -- i.e. there was a prior
/// press within `window`. First press (or press after window
/// expired) returns `false`; the caller updates the
/// `Option<Instant>` it threads through the SIGINT handler so the
/// NEXT press, if it lands in time, fires.
#[must_use]
pub fn should_double_cancel(
    previous_press: Option<Instant>,
    now: Instant,
    window: Duration,
) -> bool {
    match previous_press {
        Some(prev) => now.duration_since(prev) <= window,
        None => false,
    }
}

/// Saga 21.5 step 003: POST `/v1/sessions/<id>/cancel`. Used by
/// the REPL's SIGINT debounce path when the user double-taps
/// Ctrl-C. Returns `Ok(())` on `200`; idempotent at the server
/// level so calling twice is safe.
pub fn post_cancel(
    client: &reqwest::blocking::Client,
    base_url: &str,
    session_id: &str,
    token: &str,
) -> Result<(), ClientError> {
    let resp = client
        .post(format!(
            "{}/v1/sessions/{session_id}/cancel",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(token)
        .send()
        .map_err(|e| ClientError::Network(e.to_string()))?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
            .unwrap_or(body);
        return Err(ClientError::Server { status, message });
    }
    Ok(())
}

/// POST `/v1/sessions/<id>/eval_stream` and consume the
/// SSE body line-by-line. `on_metric` fires once per
/// `event: metric` frame; the function returns when the
/// stream emits a terminal `event: done` (Ok) or
/// `event: error` (Err with `ClientError::Server` mapped
/// to HTTP 400 to match the non-streaming `/eval` error
/// taxonomy). Transport-level failures map to
/// `ClientError::Network`.
pub fn eval_remote_stream(
    client: &reqwest::blocking::Client,
    base_url: &str,
    session_id: &str,
    token: &str,
    program: &str,
    on_metric: &mut dyn FnMut(&SseMetric),
) -> Result<EvalResponse, ClientError> {
    let resp = client
        .post(format!(
            "{}/v1/sessions/{session_id}/eval_stream",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(token)
        .json(&serde_json::json!({"program": program}))
        .send()
        .map_err(|e| ClientError::Network(e.to_string()))?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
            .unwrap_or(body);
        return Err(ClientError::Server { status, message });
    }
    consume_sse_stream(BufReader::new(resp), on_metric)
}

/// Consume the framed SSE body. Separated from
/// `eval_remote_stream` so the wire setup and the parser
/// each stay under the sw-checklist 25-line LOC warning
/// budget.
fn consume_sse_stream<R: BufRead>(
    reader: R,
    on_metric: &mut dyn FnMut(&SseMetric),
) -> Result<EvalResponse, ClientError> {
    let mut event: Option<String> = None;
    let mut data: Option<String> = None;
    for line in reader.lines() {
        let line = line.map_err(|e| ClientError::Network(format!("stream read: {e}")))?;
        if line.is_empty() {
            if let Some(outcome) = handle_frame(event.take(), data.take(), on_metric)? {
                return Ok(outcome);
            }
        } else if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data = Some(rest.trim().to_string());
        }
    }
    Err(ClientError::Network(
        "stream ended without done/error".into(),
    ))
}

/// Dispatch one assembled SSE frame. `Ok(Some(_))` means
/// terminal (`done` -> return to caller); `Ok(None)`
/// means continue reading (`ready` / `metric`); `Err`
/// means terminal-error (the server sent `event: error`
/// or the payload failed to parse).
fn handle_frame(
    event: Option<String>,
    data: Option<String>,
    on_metric: &mut dyn FnMut(&SseMetric),
) -> Result<Option<EvalResponse>, ClientError> {
    let (Some(event), Some(data)) = (event, data) else {
        return Ok(None);
    };
    match event.as_str() {
        "ready" => Ok(None),
        "metric" => {
            let m: MetricData = serde_json::from_str(&data)
                .map_err(|e| ClientError::Network(format!("decode metric: {e}")))?;
            on_metric(&SseMetric {
                name: m.name,
                step: m.step,
                value: m.value,
            });
            Ok(None)
        }
        "done" => {
            let d: DoneData = serde_json::from_str(&data)
                .map_err(|e| ClientError::Network(format!("decode done: {e}")))?;
            Ok(Some(EvalResponse {
                value: d.value,
                kind: d.kind,
            }))
        }
        "error" => {
            let e: ErrorData = serde_json::from_str(&data)
                .map_err(|e| ClientError::Network(format!("decode error: {e}")))?;
            Err(ClientError::Server {
                status: 400,
                message: e.error,
            })
        }
        "cancelled" => {
            let c: CancelledData = serde_json::from_str(&data)
                .map_err(|e| ClientError::Network(format!("decode cancelled: {e}")))?;
            Err(ClientError::Cancelled {
                step: c.step,
                partial_losses: c.partial_losses,
            })
        }
        _ => Ok(None),
    }
}
