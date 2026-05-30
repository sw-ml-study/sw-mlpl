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

use mlpl_runtime_core::error::RuntimeError;

use crate::llm_http::{ask_body, send_ask};

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
    call_ollama_with_system(url, prompt, model, "")
}

/// Like [`call_ollama`] but also sets Ollama's `system` field
/// (its grounding/instruction channel) when `system` is non-empty.
/// `:ask` uses this to put the "you are inside sw-MLPL" preamble +
/// session context in the system role, which weak models follow
/// far better than the same text inlined in the prompt.
pub fn call_ollama_with_system(
    url: &str,
    prompt: &str,
    model: &str,
    system: &str,
) -> Result<String, RuntimeError> {
    let resolved = resolve_url(url);
    let json = send_ask(&resolved, ask_body(prompt, model, system))?;
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
