//! HTTP transport for the `llm_call` builtin: build the
//! `/api/generate` request body, POST it, and lift connection /
//! status / decode failures into `RuntimeError`. The builtin
//! entry points + URL/response shaping live in `llm_builtins.rs`;
//! this module owns only the wire path.

use std::time::Duration;

use mlpl_runtime_core::error::RuntimeError;

/// Fail fast when Ollama is not even accepting connections.
const CONNECT_TIMEOUT_SECS: u64 = 5;
/// Generation is legitimately slow: a cold model (re)load alone can
/// take a minute on a large model, and a long grounded answer streams
/// out after that (`stream: false` means the whole reply arrives at
/// the end). 120s of TOTAL budget produced real-user "timed out
/// reading response" failures, so the read window is generous.
const RESPONSE_TIMEOUT_SECS: u64 = 600;
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
        .timeout_connect(Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(RESPONSE_TIMEOUT_SECS))
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

/// GET `{host}/api/tags` -> the installed models as `(name, size_bytes)`,
/// dropping embedding models (useless for `:ask`). Errors are returned as
/// strings (callers treat "can't list" as "no models, use the fallback").
pub fn list_models(host: &str) -> Result<Vec<(String, u64)>, String> {
    let url = format!("{}/api/tags", host.trim_end_matches('/'));
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(10))
        .build();
    let resp = agent
        .get(&url)
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;
    let v: serde_json::Value = resp.into_json().map_err(|e| format!("decode {url}: {e}"))?;
    let models = v["models"].as_array().into_iter().flatten();
    Ok(models
        .filter_map(|m| {
            let name = m["name"].as_str().unwrap_or_default();
            (!name.is_empty() && !name.contains("embed"))
                .then(|| (name.to_string(), m["size"].as_u64().unwrap_or(0)))
        })
        .collect())
}

/// Below ~2 GB (roughly a 3B-parameter model at typical quantization) a
/// model is too small/weak to give useful general `:ask` answers; the
/// median pick skips these so a cluster of toy/experimental installs does
/// not drag the default onto a weak model.
const MIN_VIABLE_MODEL_BYTES: u64 = 2_000_000_000;

/// Median-by-size name of a non-empty model slice.
fn median_by_size(models: &[&(String, u64)]) -> String {
    let mut sorted = models.to_vec();
    sorted.sort_by_key(|(_, size)| *size);
    sorted[sorted.len() / 2].0.clone()
}

/// The `:ask` default model: the median-by-size model that can actually
/// hold a conversation. Two filters before the median:
/// - embedding models are excluded outright -- they return vectors, not
///   text, so they can never answer a prompt (a large embed model would
///   otherwise win the median);
/// - models below `MIN_VIABLE_MODEL_BYTES` are skipped, UNLESS that would
///   leave nothing (a host with only small models still gets a pick).
///
/// `None` only when no chat-capable model is installed.
#[must_use]
pub fn median_model(models: &[(String, u64)]) -> Option<String> {
    let chat: Vec<&(String, u64)> = models
        .iter()
        .filter(|(name, _)| !name.contains("embed"))
        .collect();
    let viable: Vec<&(String, u64)> = chat
        .iter()
        .copied()
        .filter(|(_, size)| *size >= MIN_VIABLE_MODEL_BYTES)
        .collect();
    match (viable.is_empty(), chat.is_empty()) {
        (false, _) => Some(median_by_size(&viable)),
        (true, false) => Some(median_by_size(&chat)),
        (true, true) => None,
    }
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
