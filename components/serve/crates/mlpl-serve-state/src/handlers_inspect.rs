//! Session-introspection handlers (`/inspect`, `GET /v1/sessions/{id}`)
//! and the workspace snapshot they share. Split from `handlers.rs`
//! (connect-telemetry step 003) to bring that module back inside the
//! sw-checklist file-LOC and function-count budgets.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use mlpl_eval::Environment;
use serde::Serialize;
use uuid::Uuid;

use crate::config::AppState;
use mlpl_serve_core::eval_viz::{ErrorResponse, json_err};

const VARS_CAP: usize = 200;

#[derive(Serialize)]
pub struct VarSnapshot {
    pub name: String,
    pub shape: Vec<usize>,
    pub is_param: bool,
}

#[derive(Serialize)]
pub struct InspectResponse {
    pub vars: Vec<VarSnapshot>,
    pub models: Vec<String>,
    pub tokenizers: Vec<String>,
    pub experiments: Vec<String>,
    pub more: usize,
}

/// Saga 21.5 step 009: `GET /v1/sessions/{id}` response.
/// Superset of `InspectResponse` plus the session id and
/// creation/last-eval timestamps the reattach client uses to
/// render a "you are rejoining a session last touched N seconds
/// ago" banner.
#[derive(Serialize)]
pub struct SessionMetaResponse {
    pub session_id: Uuid,
    pub created_at: u64,
    pub last_eval_at: Option<u64>,
    pub vars: Vec<VarSnapshot>,
    pub models: Vec<String>,
    pub tokenizers: Vec<String>,
    pub experiments: Vec<String>,
    pub more: usize,
}

/// `GET /v1/sessions/{id}/inspect` -- requires
/// bearer when `auth_mode == Required`. Returns a
/// JSON snapshot of the session's workspace
/// (variable names + shapes + `[param]` tags, model
/// names, tokenizer names, experiment names). Vars
/// capped at 200 entries; the `more` field reports
/// how many were truncated.
pub async fn inspect_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Json<InspectResponse>, (StatusCode, Json<ErrorResponse>)> {
    let sessions = state.sessions.read().await;
    let session = sessions
        .get(&id)
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
    mlpl_serve_core::sessions::require_bearer(state.auth_mode, &session.token, &headers)?;
    Ok(Json(snapshot_env(&session.env)))
}

/// `GET /v1/sessions/{id}` -- requires bearer when
/// `auth_mode == Required`. Returns the session's bearer-token
/// signed metadata: creation + last-eval timestamps plus the
/// same workspace summary `/inspect` returns. Saga 21.5 step
/// 009 -- backs `mlpl-repl --connect --session <id>`.
pub async fn session_meta_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Json<SessionMetaResponse>, (StatusCode, Json<ErrorResponse>)> {
    let sessions = state.sessions.read().await;
    let session = sessions
        .get(&id)
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
    mlpl_serve_core::sessions::require_bearer(state.auth_mode, &session.token, &headers)?;
    let snap = snapshot_env(&session.env);
    Ok(Json(SessionMetaResponse {
        session_id: id,
        created_at: session.created_at,
        last_eval_at: session.last_eval_at,
        vars: snap.vars,
        models: snap.models,
        tokenizers: snap.tokenizers,
        experiments: snap.experiments,
        more: snap.more,
    }))
}

pub(crate) fn snapshot_env(env: &Environment) -> InspectResponse {
    let mut vars: Vec<VarSnapshot> = env
        .vars_iter()
        .map(|(name, arr)| VarSnapshot {
            name: name.clone(),
            shape: arr.shape().dims().to_vec(),
            is_param: env.is_param(name),
        })
        .collect();
    vars.sort_by(|a, b| a.name.cmp(&b.name));
    let total = vars.len();
    let more = total.saturating_sub(VARS_CAP);
    vars.truncate(VARS_CAP);
    let mut models: Vec<String> = env.models_iter().map(|(n, _)| n.clone()).collect();
    models.sort();
    let mut tokenizers: Vec<String> = env.tokenizers_iter().map(|(n, _)| n.clone()).collect();
    tokenizers.sort();
    let mut experiments: Vec<String> = env
        .experiment_log()
        .iter()
        .map(|r| r.name.clone())
        .collect();
    experiments.sort();
    experiments.dedup();
    InspectResponse {
        vars,
        models,
        tokenizers,
        experiments,
        more,
    }
}

/// Whether `host` answers `GET /api/tags` within a short timeout.
/// Any failure (refused, timeout, non-2xx) is "not alive" -- the
/// probe must never stall the `/v1/devices` response noticeably.
#[must_use]
pub fn ollama_alive(host: &str) -> bool {
    let url = format!("{}/api/tags", host.trim_end_matches('/'));
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(800))
        .build()
        .ok()
        .and_then(|c| c.get(url).send().ok())
        .is_some_and(|r| r.status().is_success())
}

/// Stateful `GET /v1/devices`: compiled devices + hostname + a LIVE
/// Ollama liveness verdict against this server's configured host
/// (blocking reqwest on a blocking thread, per project convention).
/// The web UI gates the "Ask Ollama" demo on the flag.
pub async fn devices_handler(State(state): State<AppState>) -> Json<impl Serialize> {
    let host = state.ollama.default_host.clone();
    let alive = tokio::task::spawn_blocking(move || ollama_alive(&host))
        .await
        .unwrap_or(false);
    Json(mlpl_serve_core::devices::devices_response(alive))
}
