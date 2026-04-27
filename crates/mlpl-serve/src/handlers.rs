//! HTTP route handlers. Saga 21 step 001.
//!
//! Each handler is a thin axum extractor wrapper
//! around a small piece of business logic in
//! `sessions` or the eval pipeline. Auth is enforced
//! by middleware in `server::build_app`; handlers
//! assume a request that reaches them is already
//! authorized (when auth is required).

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::auth::{AuthMode, check_token, extract_bearer};
use crate::server::AppState;

const VARS_CAP: usize = 200;

#[derive(Serialize)]
pub struct CreateSessionResponse {
    pub session_id: Uuid,
    pub token: String,
}

#[derive(Deserialize)]
pub struct EvalRequest {
    pub program: String,
}

#[derive(Serialize)]
pub struct EvalResponse {
    pub value: String,
    pub kind: &'static str,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

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

/// `POST /v1/sessions` -- no auth. Creates a fresh
/// session and returns its id + bearer token.
pub async fn create_session_handler(State(state): State<AppState>) -> impl IntoResponse {
    let (id, token) = crate::sessions::create_session(&state.sessions).await;
    (
        StatusCode::OK,
        Json(CreateSessionResponse {
            session_id: id,
            token,
        }),
    )
}

/// `POST /v1/sessions/{id}/eval` -- requires bearer
/// when `auth_mode == Required`. Lex + parse + run
/// the program against the session's env, return
/// the stringified value + kind.
pub async fn eval_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(body): Json<EvalRequest>,
) -> Result<Json<EvalResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut sessions = state.sessions.write().await;
    let session = sessions
        .get_mut(&id)
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
    if state.auth_mode == AuthMode::Required {
        let provided = extract_bearer(&headers).ok_or((
            StatusCode::UNAUTHORIZED,
            json_err("missing or invalid authorization"),
        ))?;
        if !check_token(provided, &session.token) {
            return Err((
                StatusCode::UNAUTHORIZED,
                json_err("missing or invalid authorization"),
            ));
        }
    }
    let tokens =
        lex(&body.program).map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e:?}"))))?;
    let stmts =
        parse(&tokens).map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e:?}"))))?;
    session
        .env
        .set_peer_dispatcher(Arc::new(crate::server::RemoteMlxDispatcher::new(
            state.peers.clone(),
            state.peer_sessions.clone(),
        )));
    let value = eval_program_value(&stmts, &mut session.env);
    session.env.clear_peer_dispatcher();
    let value = value.map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e}"))))?;
    let kind = value_kind(&value);
    Ok(Json(EvalResponse {
        value: format!("{value}"),
        kind,
    }))
}

/// `GET /v1/health` -- no auth. Liveness +
/// `CARGO_PKG_VERSION`.
pub async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
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
    if state.auth_mode == AuthMode::Required {
        let provided = extract_bearer(&headers).ok_or((
            StatusCode::UNAUTHORIZED,
            json_err("missing or invalid authorization"),
        ))?;
        if !check_token(provided, &session.token) {
            return Err((
                StatusCode::UNAUTHORIZED,
                json_err("missing or invalid authorization"),
            ));
        }
    }
    Ok(Json(snapshot_env(&session.env)))
}

fn snapshot_env(env: &Environment) -> InspectResponse {
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

fn json_err(msg: impl Into<String>) -> Json<ErrorResponse> {
    Json(ErrorResponse { error: msg.into() })
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Array(_) => "array",
        Value::Str(_) => "string",
        Value::Model(_) => "model",
        Value::Tokenizer(_) => "tokenizer",
        Value::DeviceTensor { .. } => "device-tensor",
    }
}
