//! Saga 77: WASM-only HTTP helpers extracted from `eval_wasm.rs`
//! so the parent stays under the sw-checklist function count limit.
//! Connect-telemetry step 002 moved the streaming transport to its
//! own sibling (`eval_wasm_stream.rs`) and pulled the non-streaming
//! `wasm_eval` body here, split decision-from-transport.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use crate::eval::RemoteSession;

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

/// Non-streaming connect eval: ensure a session, POST `/eval`, and
/// render the JSON body as the REPL display string.
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
fn eval_body_display(program: &str, body: &serde_json::Value) -> String {
    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return format!("error: {err}");
    }
    crate::connect_viz::emit_from_response(program, body);
    body.get("value")
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("error: missing value: {body}"), str::to_string)
}
