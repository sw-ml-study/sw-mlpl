//! Native (non-wasm) `RemoteEvaluator` REST impl: the blocking
//! `/eval` path used by native tests and the `#[path]`-include
//! harnesses. Split from `eval.rs` (connect-telemetry step 006) to
//! bring that module inside the function-count budget.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;

use crate::eval::{Evaluator, RemoteEvaluator, RemoteSession, ResultCb};

impl Evaluator for RemoteEvaluator {
    fn eval(&self, program: &str, on_result: ResultCb) {
        let result = native_eval(self.base_url(), &self.state_handle(), program);
        on_result(result);
    }
    fn clear(&self) {
        self.clear_state();
    }
}

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
fn body_value(body: &serde_json::Value) -> String {
    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return format!("error: {err}");
    }
    body.get("value")
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("error: missing value: {body}"), str::to_string)
}

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
