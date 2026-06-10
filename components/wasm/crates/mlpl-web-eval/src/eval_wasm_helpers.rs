//! Saga 77: WASM-only HTTP helpers extracted from `eval_wasm.rs`
//! so the parent stays under the sw-checklist function count limit.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use crate::eval::{MetricCb, RemoteSession, StreamOutcome};
use crate::eval_sse::parse_sse_stream;

/// Race a connect-server request against a timeout so a wedged or absent
/// `mlpl-serve` fails FAST with a clear message instead of hanging the
/// REPL (the connect-only `:ask` / CUDA / MLX demos all go through here).
pub(crate) async fn with_deadline<T>(
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

pub(crate) async fn wasm_create_session(base_url: &str) -> Result<RemoteSession, String> {
    let url = format!("{}/v1/sessions", base_url.trim_end_matches('/'));
    // Session creation is trivial -- a slow/no response here means the
    // backend is down or wedged, so a short deadline fails fast.
    let resp = with_deadline(6000, "connect", async {
        gloo::net::http::Request::post(&url)
            .send()
            .await
            .map_err(|e| e.to_string())
    })
    .await?;
    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let session_id = body
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or("missing session_id")?
        .to_string();
    let token = body
        .get("token")
        .and_then(|v| v.as_str())
        .ok_or("missing token")?
        .to_string();
    Ok(RemoteSession { session_id, token })
}

pub(crate) async fn wasm_eval_stream(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
    mut on_metric: MetricCb,
) -> StreamOutcome {
    match wasm_eval_stream_inner(base_url, state, program, &mut on_metric).await {
        Ok(outcome) => outcome,
        Err(message) => StreamOutcome::Error { message },
    }
}

// Inner helper returning `Result<StreamOutcome, String>` so the
// `?` operator can collapse the half-dozen fallible
// `gloo::net::http` steps into one error path. Browser fetch
// responses don't expose a Rust `BufRead`, so we pull the whole
// body as a string and feed it through the shared SSE parser --
// this forfeits true live streaming on WASM (frames batch at
// body end). A follow-up step will switch to ReadableStream
// chunk reads via web-sys.
async fn wasm_eval_stream_inner(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
    on_metric: &mut MetricCb,
) -> Result<StreamOutcome, String> {
    if state.borrow().is_none() {
        let s = wasm_create_session(base_url).await?;
        *state.borrow_mut() = Some(s);
    }
    let s = state.borrow().as_ref().expect("session").clone();
    let url = format!(
        "{}/v1/sessions/{}/eval_stream",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    let req = gloo::net::http::Request::post(&url)
        .header("Authorization", &format!("Bearer {}", s.token))
        .header("Content-Type", "application/json")
        .body(serde_json::json!({"program": program}).to_string())
        .map_err(|e| e.to_string())?;
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Ok(StreamOutcome::Error {
            message: resp.text().await.unwrap_or_default(),
        });
    }
    let text = resp.text().await.unwrap_or_default();
    Ok(parse_sse_stream(std::io::Cursor::new(text), on_metric))
}
