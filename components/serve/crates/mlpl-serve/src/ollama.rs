//! HTTP endpoints exposing the server's Ollama settings: report the
//! configured default host + model, and proxy the model list from
//! the configured host's `/api/tags`. The browser cannot reach
//! Ollama directly (CORS + it may be a LAN host), so the server
//! fetches on its behalf -- only for allow-listed hosts. Phase 0 of
//! the local-gpu-agentic saga.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::server::AppState;

/// Optional `?host=` override for `GET /v1/ollama/tags`.
#[derive(Deserialize)]
pub struct TagsQuery {
    pub host: Option<String>,
}

/// `GET /v1/ollama/config` -> the server-configured default host +
/// model, so the web `:ask` can default to them without carrying
/// the host/model in the page URL. The model is the configured default
/// when it is actually installed, otherwise a median-size installed
/// model (so `:ask` never 404s on a missing default). Override the
/// configured default with `mlpl-serve --ollama-model <name>`.
pub async fn config_handler(State(state): State<AppState>) -> Json<Value> {
    let host = state.ollama.default_host.clone();
    let configured = state.ollama.default_model.clone();
    let model = tokio::task::spawn_blocking(move || resolve_model(&host, &configured))
        .await
        .unwrap_or_else(|_| state.ollama.default_model.clone());
    Json(json!({ "host": state.ollama.default_host, "model": model }))
}

/// The effective `:ask` model for `host`: the `configured` model if it is
/// installed there, else the shared `median_model` pick (a median-by-size
/// chat-capable model -- embedding models and toy/sub-viable models are
/// skipped), else `configured` (Ollama unreachable / no chat model).
fn resolve_model(host: &str, configured: &str) -> String {
    let Ok(tags) = fetch_tags(host) else {
        return configured.to_string();
    };
    let models: Vec<(String, u64)> = tags["models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let n = m["name"].as_str().unwrap_or_default();
            (!n.is_empty()).then(|| (n.to_string(), m["size"].as_u64().unwrap_or(0)))
        })
        .collect();
    if models.iter().any(|(n, _)| n == configured) {
        return configured.to_string();
    }
    mlpl_runtime::median_model(&models).unwrap_or_else(|| configured.to_string())
}

/// `GET /v1/ollama/tags?host=<optional>` -> the Ollama model list
/// from `<host>/api/tags`. `host` defaults to the configured
/// default and MUST be allow-listed; a disallowed host is 403, an
/// unreachable/!2xx host is 502.
pub async fn tags_handler(
    State(state): State<AppState>,
    Query(q): Query<TagsQuery>,
) -> Result<Json<Value>, StatusCode> {
    let host = q.host.unwrap_or_else(|| state.ollama.default_host.clone());
    if !state.ollama.is_allowed(&host) {
        return Err(StatusCode::FORBIDDEN);
    }
    let body = tokio::task::spawn_blocking(move || fetch_tags(&host))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    Ok(Json(body))
}

/// Blocking `GET <host>/api/tags` on a blocking thread (reqwest is
/// the project's blocking client). Returns the parsed JSON body.
fn fetch_tags(host: &str) -> Result<Value, String> {
    let url = format!("{}/api/tags", host.trim_end_matches('/'));
    reqwest::blocking::Client::new()
        .get(url)
        .send()
        .map_err(|e| e.to_string())?
        .json::<Value>()
        .map_err(|e| e.to_string())
}
