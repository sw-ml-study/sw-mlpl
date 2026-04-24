//! Saga 19 step 001: `llm_call` POSTs to an Ollama-
//! compatible `/api/generate` endpoint and returns
//! the model's completion text.
//!
//! This module owns ONLY the pure HTTP path: no
//! `Expr` / `Environment` / arity-validation logic
//! lives here -- those are the eval-side dispatcher's
//! job. The split mirrors Saga 22's revised layout
//! (pure-IO helpers in `mlpl-runtime`, `Expr`-aware
//! shims in `mlpl-eval`).

use std::time::Duration;

use crate::error::RuntimeError;

const TIMEOUT_SECS: u64 = 120;
const BODY_PREVIEW_CHARS: usize = 200;

/// POST `prompt` to an Ollama-compatible
/// `/api/generate` endpoint at `url` and return the
/// model's completion text. The URL is normalized
/// via [`resolve_url`]; trailing slashes are stripped
/// and `/api/generate` is appended unless already
/// present.
///
/// # Errors
/// Returns [`RuntimeError::InvalidArgument`] (with
/// `func = "llm_call"`) for connection failures,
/// non-2xx status codes, invalid response JSON, or
/// missing `response` field. The eval-side
/// dispatcher lifts this into an `EvalError` for
/// MLPL surface error reporting.
pub fn call_ollama(url: &str, prompt: &str, model: &str) -> Result<String, RuntimeError> {
    let resolved = resolve_url(url);
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
    });
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .build();
    let resp = match agent
        .post(&resolved)
        .set("Content-Type", "application/json")
        .send_json(body)
    {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let preview: String = r
                .into_string()
                .unwrap_or_default()
                .chars()
                .take(BODY_PREVIEW_CHARS)
                .collect();
            return Err(RuntimeError::InvalidArgument {
                func: "llm_call".into(),
                reason: format!("POST {resolved} returned {code}: {preview}"),
            });
        }
        Err(e) => {
            return Err(RuntimeError::InvalidArgument {
                func: "llm_call".into(),
                reason: format!("POST {resolved} failed: {e}"),
            });
        }
    };
    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| RuntimeError::InvalidArgument {
            func: "llm_call".into(),
            reason: format!("invalid JSON from {resolved}: {e}"),
        })?;
    parse_response(&json)
}

/// Normalize the user-supplied URL: strip trailing
/// slashes, then append `/api/generate` unless the
/// URL already ends with it.
fn resolve_url(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    if trimmed.ends_with("/api/generate") {
        trimmed.into()
    } else {
        format!("{trimmed}/api/generate")
    }
}

/// Pull the top-level `response` string field out of
/// an Ollama `/api/generate` reply.
fn parse_response(json: &serde_json::Value) -> Result<String, RuntimeError> {
    json.get("response")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| RuntimeError::InvalidArgument {
            func: "llm_call".into(),
            reason: format!("response JSON missing string `response` field: {json}"),
        })
}
