//! Saga 21 step 002: HTTP client for `mlpl-serve`.
//! Pure transport -- the REPL loop, slash dispatch,
//! and `--connect` argv parsing live in
//! `connect_repl.rs`.
//!
//! `:ask` keeps using `mlpl_runtime::call_ollama`
//! directly (see `connect_repl::dispatch_slash`); the
//! local `OLLAMA_HOST` / `OLLAMA_MODEL` env vars
//! still override.

use std::time::Duration;

use serde::Deserialize;

const TIMEOUT_SECS: u64 = 120;

/// Errors returned by the connect-mode HTTP path.
/// Surfaced to stderr; the REPL keeps reading after
/// each one so a transient server error doesn't kill
/// the session.
#[derive(Debug)]
pub enum ClientError {
    Network(String),
    Server { status: u16, message: String },
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(m) => write!(f, "network error: {m}"),
            Self::Server { status, message } => write!(f, "server {status}: {message}"),
        }
    }
}

#[derive(Deserialize)]
struct CreateSessionResponse {
    session_id: String,
    token: String,
}

#[derive(Deserialize, Debug)]
pub struct EvalResponse {
    pub value: String,
    /// Tests inspect this; the REPL just prints
    /// `value`. A future viz / formatting step may
    /// switch on `kind`, so it stays in the
    /// deserialized shape.
    #[allow(dead_code)]
    pub kind: String,
}

#[derive(Deserialize, Debug)]
pub struct VarSnapshot {
    pub name: String,
    pub shape: Vec<usize>,
    pub is_param: bool,
}

#[derive(Deserialize, Debug)]
pub struct InspectResponse {
    pub vars: Vec<VarSnapshot>,
    pub models: Vec<String>,
    pub tokenizers: Vec<String>,
    pub experiments: Vec<String>,
    pub more: usize,
}

/// Build the long-lived blocking client used for
/// every call in a connect-mode session. 120-second
/// timeout matches `:ask` and the language-level
/// `llm_call`.
pub fn build_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .build()
        .expect("blocking reqwest client builds")
}

/// POST `<base>/v1/sessions` with no auth. Returns
/// the (session_id, token) pair to use for every
/// subsequent call.
pub fn create_session(
    client: &reqwest::blocking::Client,
    base_url: &str,
) -> Result<(String, String), ClientError> {
    let resp = client
        .post(format!("{}/v1/sessions", base_url.trim_end_matches('/')))
        .send()
        .map_err(|e| ClientError::Network(e.to_string()))?;
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        return Err(ClientError::Server {
            status,
            message: body,
        });
    }
    let body: CreateSessionResponse = resp
        .json()
        .map_err(|e| ClientError::Network(format!("decode: {e}")))?;
    Ok((body.session_id, body.token))
}

/// POST `<base>/v1/sessions/{id}/eval` with the
/// bearer token; returns the server's `{value, kind}`
/// or a `Server` error carrying the unwrapped
/// `error` string for surface-level errors (parse,
/// eval, auth).
pub fn eval_remote(
    client: &reqwest::blocking::Client,
    base_url: &str,
    session_id: &str,
    token: &str,
    program: &str,
) -> Result<EvalResponse, ClientError> {
    let resp = client
        .post(format!(
            "{}/v1/sessions/{session_id}/eval",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(token)
        .json(&serde_json::json!({"program": program}))
        .send()
        .map_err(|e| ClientError::Network(e.to_string()))?;
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
            .unwrap_or(body);
        return Err(ClientError::Server { status, message });
    }
    resp.json()
        .map_err(|e| ClientError::Network(format!("decode: {e}")))
}

/// GET `<base>/v1/sessions/{id}/inspect`. Returns
/// the structured workspace snapshot for client-side
/// slash-command rendering.
pub fn inspect_remote(
    client: &reqwest::blocking::Client,
    base_url: &str,
    session_id: &str,
    token: &str,
) -> Result<InspectResponse, ClientError> {
    let resp = client
        .get(format!(
            "{}/v1/sessions/{session_id}/inspect",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(token)
        .send()
        .map_err(|e| ClientError::Network(e.to_string()))?;
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
            .unwrap_or(body);
        return Err(ClientError::Server { status, message });
    }
    resp.json()
        .map_err(|e| ClientError::Network(format!("decode: {e}")))
}
