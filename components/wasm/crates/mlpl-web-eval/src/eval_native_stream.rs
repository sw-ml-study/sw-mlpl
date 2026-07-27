//! Native streaming (`/eval_stream`), cancel, and viz-fetch impls for
//! `RemoteEvaluator`. Split from `eval.rs` (connect-telemetry step
//! 006) alongside `eval_native.rs`.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;

use crate::eval::{
    MetricCb, RemoteEvaluator, RemoteSession, StreamCb, StreamOutcome, resolve_viz_url,
};

/// Saga 21.5 step 007: cheap-to-clone Send-able handle for
/// firing a cancel POST from a different thread (native test).
/// WASM cancels in-tick on the single-threaded JS event loop, so
/// the struct is not compiled there (it would be dead code).
#[derive(Clone)]
pub struct CancelHandle {
    base_url: String,
    session_id: String,
    token: String,
}

impl CancelHandle {
    /// Best-effort POST `/v1/sessions/<id>/cancel`. Errors are
    /// swallowed because cancel-from-side-thread inherently
    /// races with the in-flight stream; the eval result already
    /// surfaces via `StreamOutcome::Cancelled`.
    pub fn cancel(&self) {
        let url = format!(
            "{}/v1/sessions/{}/cancel",
            self.base_url.trim_end_matches('/'),
            self.session_id
        );
        let _ = reqwest::blocking::Client::new()
            .post(&url)
            .bearer_auth(&self.token)
            .send();
    }
}

impl RemoteEvaluator {
    /// Native streaming-eval impl. POSTs to `/eval_stream`, reads
    /// the SSE body line-by-line, fires `on_metric` per metric
    /// frame, and fires `on_result` exactly once with the
    /// terminal outcome.
    pub fn eval_stream(&self, program: &str, on_metric: MetricCb, on_result: StreamCb) {
        let outcome = native_eval_stream(self.base_url(), &self.state_handle(), program, on_metric);
        on_result(outcome);
    }

    /// In-tick cancel for the browser UI (single-threaded JS
    /// event loop). Tests cross threads via `cancel_handle()`
    /// instead. No-op when no session has been minted yet.
    pub fn cancel(&self) {
        if let Some(h) = self.cancel_handle() {
            h.cancel();
        }
    }

    /// Snapshot a `Send`-able handle that can fire `cancel`
    /// from a different thread. `None` until the first eval
    /// has minted a server session.
    pub fn cancel_handle(&self) -> Option<CancelHandle> {
        let session_id = self.current_session_id()?;
        let token = self.current_token()?;
        Some(CancelHandle {
            base_url: self.base_url().to_string(),
            session_id,
            token,
        })
    }

    /// Saga 21.5 step 008: native viz fetch. `GET /v1/viz/<id>`
    /// with bearer auth, returns the raw bytes + content type so
    /// callers can render an `<img>` (for `image/*`) or an
    /// `<iframe srcdoc>` (for `text/html`). Accepts both bare
    /// paths and absolute URLs. The WASM equivalent
    /// (`fetch_viz_async`) lives in `eval_wasm.rs`.
    pub fn fetch_viz(&self, viz_url: &str, bearer: &str) -> Result<(Vec<u8>, String), String> {
        let url = resolve_viz_url(self.base_url(), viz_url);
        let resp = reqwest::blocking::Client::new()
            .get(&url)
            .bearer_auth(bearer)
            .send()
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("fetch_viz: {}", resp.status()));
        }
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = resp.bytes().map_err(|e| e.to_string())?.to_vec();
        Ok((bytes, content_type))
    }
}

fn native_eval_stream(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
    mut on_metric: MetricCb,
) -> StreamOutcome {
    if state.borrow().is_none() {
        match crate::eval_wasm_helpers::native_create_session(base_url) {
            Ok(s) => *state.borrow_mut() = Some(s),
            Err(e) => return StreamOutcome::Error { message: e },
        }
    }
    let s = state.borrow().as_ref().expect("session created").clone();
    match post_eval_stream(base_url, &s, program) {
        Ok(resp) => {
            let reader = std::io::BufReader::new(resp);
            crate::eval_sse::parse_sse_stream(reader, &mut on_metric)
        }
        Err(message) => StreamOutcome::Error { message },
    }
}

/// POST `/eval_stream`; a non-2xx body collapses to its JSON `error`
/// field (or the raw text) as the `Err` message.
fn post_eval_stream(
    base_url: &str,
    s: &RemoteSession,
    program: &str,
) -> Result<reqwest::blocking::Response, String> {
    let url = format!(
        "{}/v1/sessions/{}/eval_stream",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    let resp = reqwest::blocking::Client::new()
        .post(&url)
        .bearer_auth(&s.token)
        .json(&serde_json::json!({"program": program}))
        .send()
        .map_err(|e| e.to_string())?;
    if resp.status().is_success() {
        return Ok(resp);
    }
    let body = resp.text().unwrap_or_default();
    Err(serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
        .unwrap_or(body))
}
