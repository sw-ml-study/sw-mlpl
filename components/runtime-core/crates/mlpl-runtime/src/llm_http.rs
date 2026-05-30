//! HTTP transport for the `llm_call` builtin: build the
//! `/api/generate` request body, POST it, and lift connection /
//! status / decode failures into `RuntimeError`. The builtin
//! entry points + URL/response shaping live in `llm_builtins.rs`;
//! this module owns only the wire path.

use std::time::Duration;

use mlpl_runtime_core::error::RuntimeError;

const TIMEOUT_SECS: u64 = 120;
const BODY_PREVIEW_CHARS: usize = 200;

/// Build the `/api/generate` request body, adding the optional
/// `system` (grounding) field only when it is non-empty.
pub(crate) fn ask_body(prompt: &str, model: &str, system: &str) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
    });
    if !system.is_empty() {
        body["system"] = serde_json::Value::String(system.to_string());
    }
    body
}

/// POST `body` to the resolved endpoint and decode the JSON
/// reply. Connection/status/decode failures are lifted into
/// `RuntimeError::InvalidArgument` for MLPL surface reporting.
pub(crate) fn send_ask(
    resolved: &str,
    body: serde_json::Value,
) -> Result<serde_json::Value, RuntimeError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .build();
    let resp = match agent
        .post(resolved)
        .set("Content-Type", "application/json")
        .send_json(body)
    {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => return Err(http_status_error(resolved, code, r)),
        Err(e) => {
            return Err(RuntimeError::InvalidArgument {
                func: "llm_call".into(),
                reason: format!("POST {resolved} failed: {e}"),
            });
        }
    };
    resp.into_json().map_err(|e| RuntimeError::InvalidArgument {
        func: "llm_call".into(),
        reason: format!("invalid JSON from {resolved}: {e}"),
    })
}

/// Build the error for a non-2xx response, with a char-safe
/// preview of the (possibly large) error body.
fn http_status_error(resolved: &str, code: u16, r: ureq::Response) -> RuntimeError {
    let preview: String = r
        .into_string()
        .unwrap_or_default()
        .chars()
        .take(BODY_PREVIEW_CHARS)
        .collect();
    RuntimeError::InvalidArgument {
        func: "llm_call".into(),
        reason: format!("POST {resolved} returned {code}: {preview}"),
    }
}
