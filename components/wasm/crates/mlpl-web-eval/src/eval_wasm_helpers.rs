//! Saga 77: WASM-only HTTP helpers extracted from `eval_wasm.rs`
//! so the parent stays under the sw-checklist function count limit.
//! Connect-telemetry step 002 moved the streaming transport to its
//! own sibling (`eval_wasm_stream.rs`) and pulled the non-streaming
//! `wasm_eval` body here, split decision-from-transport.

use std::cell::RefCell;

use crate::eval::RemoteSession;

#[cfg(target_arch = "wasm32")]
pub(crate) use mlpl_web_eval_core::wire::with_deadline;

#[cfg(target_arch = "wasm32")]
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

/// Non-streaming connect eval: ensure a session, POST `/eval`, and
/// render the JSON body as the REPL display string.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn wasm_eval(
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
    match send_eval_request(base_url, &s, program).await {
        Ok(body) => eval_body_display(program, &body),
        Err(e) => format!("error: {e}"),
    }
}

/// POST `/v1/sessions/<id>/eval` and decode the JSON body. Generous
/// deadline: a connect eval can legitimately run minutes -- a CPU
/// base-pretrain step or a GPU fine-tune. Match mlpl-serve's own 600s
/// peer-forward timeout (peers.rs PEER_TIMEOUT_SECS) so the client
/// never gives up before the server would. The short 6s
/// session-creation deadline above is the real "is the backend up?"
/// fail-fast; once a session exists, long compute is expected.
#[cfg(target_arch = "wasm32")]
async fn send_eval_request(
    base_url: &str,
    s: &RemoteSession,
    program: &str,
) -> Result<serde_json::Value, String> {
    let url = format!(
        "{}/v1/sessions/{}/eval",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    let req = gloo::net::http::Request::post(&url)
        .header("Authorization", &format!("Bearer {}", s.token))
        .header("Content-Type", "application/json")
        .body(serde_json::json!({"program": program}).to_string())
        .map_err(|e| e.to_string())?;
    let resp = with_deadline(600_000, "eval", async {
        req.send().await.map_err(|e| e.to_string())
    })
    .await?;
    resp.json().await.map_err(|e| format!("decode: {e}"))
}

/// Interpret the `/eval` response body: surface `error`, emit the 3D
/// sculpture from the viz payload (Phase 1c), and return the display
/// value.
#[cfg(target_arch = "wasm32")]
fn eval_body_display(program: &str, body: &serde_json::Value) -> String {
    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return format!("error: {err}");
    }
    crate::connect_viz::emit_from_response(program, body);
    body.get("value")
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("error: missing value: {body}"), str::to_string)
}

// ---- non-streaming NATIVE eval (spike step 015): lives beside the
// wasm helpers so both transport-helper tails share one module; the
// native side is cfg-gated off the wasm build (no reqwest there). ----

#[cfg(not(target_arch = "wasm32"))]
use crate::eval::{Evaluator, RemoteEvaluator, ResultCb};

#[cfg(not(target_arch = "wasm32"))]
impl Evaluator for RemoteEvaluator {
    fn eval(&self, program: &str, on_result: ResultCb) {
        let result = native_eval(self.base_url(), &self.state_handle(), program);
        on_result(result);
    }
    fn clear(&self) {
        self.clear_state();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_eval(base_url: &str, state: &RefCell<Option<RemoteSession>>, program: &str) -> String {
    if state.borrow().is_none() {
        match native_create_session(base_url) {
            Ok(s) => *state.borrow_mut() = Some(s),
            Err(e) => return format!("error: {e}"),
        }
    }
    let s = state
        .borrow()
        .as_ref()
        .expect("session created above")
        .clone();
    match post_eval(base_url, &s, program) {
        Ok(body) => body_value(&body),
        Err(e) => format!("error: {e}"),
    }
}

/// POST `/v1/sessions/<id>/eval` and decode the JSON body.
#[cfg(not(target_arch = "wasm32"))]
fn post_eval(
    base_url: &str,
    s: &RemoteSession,
    program: &str,
) -> Result<serde_json::Value, String> {
    let url = format!(
        "{}/v1/sessions/{}/eval",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    reqwest::blocking::Client::new()
        .post(&url)
        .bearer_auth(&s.token)
        .json(&serde_json::json!({"program": program}))
        .send()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| format!("decode: {e}"))
}

/// The `/eval` body -> display string (an `error` field wins).
#[cfg(not(target_arch = "wasm32"))]
fn body_value(body: &serde_json::Value) -> String {
    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return format!("error: {err}");
    }
    body.get("value")
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("error: missing value: {body}"), str::to_string)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_create_session(base_url: &str) -> Result<RemoteSession, String> {
    let url = format!("{}/v1/sessions", base_url.trim_end_matches('/'));
    let body: serde_json::Value = reqwest::blocking::Client::new()
        .post(&url)
        .send()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())?;
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
