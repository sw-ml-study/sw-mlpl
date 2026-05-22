//! Saga 21.5 step 008: browser-only `RemoteEvaluator` impls.
//!
//! Lifted out of `eval.rs` so the parent module stays under its
//! 500-line file budget. Every fn is `#[cfg(target_arch =
//! "wasm32")]` because the entire module body would otherwise
//! force `gloo::net` + `wasm_bindgen_futures` on native builds.
//! `cargo test` compiles this file but elides every function;
//! the WASM build (`scripts/build-pages.sh`) wires them in.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use crate::eval::{
    Evaluator, MetricCb, RemoteEvaluator, RemoteSession, ResultCb, StreamCb, StreamOutcome,
};
use crate::eval_sse::parse_sse_stream;

impl Evaluator for RemoteEvaluator {
    fn eval(&self, program: &str, on_result: ResultCb) {
        let base = self.base_url().to_string();
        let state = self.state_handle();
        let program = program.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let result = wasm_eval(&base, &state, &program).await;
            on_result(result);
        });
    }
    fn clear(&self) {
        self.clear_state();
    }
}

async fn wasm_eval(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
) -> String {
    if state.borrow().is_none() {
        match wasm_create_session(base_url).await {
            Ok(s) => *state.borrow_mut() = Some(s),
            Err(e) => return format!("error: {e}"),
        }
    }
    let s = state
        .borrow()
        .as_ref()
        .expect("session created above")
        .clone();
    let url = format!(
        "{}/v1/sessions/{}/eval",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    let req = match gloo::net::http::Request::post(&url)
        .header("Authorization", &format!("Bearer {}", s.token))
        .header("Content-Type", "application/json")
        .body(serde_json::json!({"program": program}).to_string())
    {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let body: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => return format!("error: decode: {e}"),
    };
    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return format!("error: {err}");
    }
    body.get("value")
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("error: missing value: {body}"), str::to_string)
}

async fn wasm_create_session(base_url: &str) -> Result<RemoteSession, String> {
    let url = format!("{}/v1/sessions", base_url.trim_end_matches('/'));
    let resp = gloo::net::http::Request::post(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
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

impl RemoteEvaluator {
    /// Browser streaming-eval. Spawns a local future that POSTs
    /// to `/eval_stream`, reads the response body as a UTF-8
    /// stream, parses SSE frames, fires `on_metric` per metric,
    /// and finally fires `on_result` with the terminal outcome.
    pub fn eval_stream(&self, program: &str, on_metric: MetricCb, on_result: StreamCb) {
        let base = self.base_url().to_string();
        let state = self.state_handle();
        let program = program.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let outcome = wasm_eval_stream(&base, &state, &program, on_metric).await;
            on_result(outcome);
        });
    }

    /// Browser cancel: fire `/v1/sessions/<id>/cancel`. No-op
    /// when no session has been minted yet.
    pub fn cancel(&self) {
        let Some(session_id) = self.current_session_id() else {
            return;
        };
        let Some(token) = self.current_token() else {
            return;
        };
        let url = format!(
            "{}/v1/sessions/{}/cancel",
            self.base_url().trim_end_matches('/'),
            session_id
        );
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(req) = gloo::net::http::Request::post(&url)
                .header("Authorization", &format!("Bearer {token}"))
                .body(String::new())
            {
                let _ = req.send().await;
            }
        });
    }

    /// Browser viz fetch: `GET /v1/viz/<id>` with bearer auth.
    /// Returns `(bytes, content_type)` so the caller can render
    /// an `<img>` (for `image/*`) or an `<iframe srcdoc>` (for
    /// `text/html`). Native equivalent lives in `eval.rs`.
    pub async fn fetch_viz_async(
        &self,
        viz_url: &str,
        bearer: &str,
    ) -> Result<(Vec<u8>, String), String> {
        let url = crate::eval::resolve_viz_url(self.base_url(), viz_url);
        let resp = gloo::net::http::Request::get(&url)
            .header("Authorization", &format!("Bearer {bearer}"))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.ok() {
            return Err(format!("fetch_viz: {}", resp.status()));
        }
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let bytes = resp.binary().await.map_err(|e| e.to_string())?;
        Ok((bytes, content_type))
    }
}

async fn wasm_eval_stream(
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
