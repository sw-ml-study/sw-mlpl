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
    /// Saga 21.5 step 004: when the eval returned an SVG-shaped
    /// string, the server stashes the bytes in the `viz_storage`
    /// content-addressed store and surfaces the URL here.
    /// `None` (skipped on serialization) for non-SVG results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viz_url: Option<String>,
    /// Saga 21.5 step 004: server-side path inside
    /// `MLPL_CACHE_DIR` (when set) where the same SVG was also
    /// written. The dev-loopback case (`mlpl-serve` and
    /// `mlpl-repl` on one host) lets the client open the file
    /// directly; absent when no cache dir is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viz_local_path: Option<String>,
    /// Phase 1c: viz data so the connect-mode web UI can emit a 3D
    /// sculpture for a server-evaluated result. `shape` empty +
    /// `values`/`string_list` absent for non-array/list results.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub shape: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub string_list: Option<Vec<String>>,
    /// Phase 1c part 2: a model's Sankey decomposition, so a model
    /// evaluated in connect mode renders its diagram. `None` for
    /// non-model results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viz_node: Option<mlpl_web_viz_ir::VizNode>,
}

#[derive(Serialize)]
pub struct CancelResponse {
    pub cancelled: bool,
}

#[derive(Serialize)]
pub struct ResetResponse {
    pub cancelled: usize,
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

/// `POST /v1/sessions` -- no auth. Creates a fresh
/// session and returns its id + bearer token. Also
/// registers the session in the parallel interrupt
/// map (Saga 21.5 step 003) so `/cancel` is reachable
/// for it from the very first call.
pub async fn create_session_handler(State(state): State<AppState>) -> impl IntoResponse {
    let (id, token) = crate::sessions::create_session(&state.sessions, &state.interrupts).await;
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
    install_session_interrupt(&state, &id, session).await;
    session
        .env
        .set_peer_dispatcher(Arc::new(crate::server::RemoteMlxDispatcher::new(
            state.peers.clone(),
            state.peer_sessions.clone(),
        )));
    let value = eval_program_value(&stmts, &mut session.env);
    session.env.clear_peer_dispatcher();
    session.env.clear_interrupt();
    let value = value.map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e}"))))?;
    let kind = value_kind(&value);
    let formatted = format!("{value}");
    session.last_eval_at = Some(crate::sessions::now_unix_seconds());
    let attached = crate::viz_storage::attach_viz(&state.viz, &formatted, kind).await;
    drop(sessions);
    crate::persist::maybe_flush(&state).await;
    Ok(Json(crate::eval_viz::build_eval_response(
        &value, kind, formatted, attached,
    )))
}

/// `POST /v1/sessions/{id}/cancel` -- requires bearer
/// when `auth_mode == Required`. Flips the session's
/// shared `Interrupt` bool so any in-flight eval
/// observes the trip at its next loop / pre-builtin
/// checkpoint and raises `EvalError::Cancelled`. Idempotent:
/// a second call after the bool is already set is a no-op
/// at the eval level. Uses the parallel `InterruptMap`
/// (Saga 21.5 step 003) so it can run while another
/// request holds the sessions write lock. Saga 21.5
/// step 003.
pub async fn cancel_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Json<CancelResponse>, (StatusCode, Json<ErrorResponse>)> {
    let not_found = || (StatusCode::NOT_FOUND, json_err("unknown session"));
    let unauthorized = || {
        (
            StatusCode::UNAUTHORIZED,
            json_err("missing or invalid authorization"),
        )
    };
    let entry = state
        .interrupts
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(not_found)?;
    if state.auth_mode == AuthMode::Required {
        let provided = extract_bearer(&headers).ok_or_else(unauthorized)?;
        if !check_token(provided, &entry.token) {
            return Err(unauthorized());
        }
    }
    entry.interrupt.set();
    Ok(Json(CancelResponse { cancelled: true }))
}

/// `POST /v1/reset` -- no auth, no session id. Sets the interrupt flag
/// on EVERY registered session so any in-flight eval (notably a
/// training loop) aborts at its next checkpoint and releases the
/// sessions write-lock. This is the UI's recovery path when it has lost
/// its session id+token (e.g. after a shift-reload during a heavy
/// demo): a fresh page can free the backend without a process restart.
/// It deliberately needs no bearer (the reloaded page has none) and
/// uses the lock-free interrupt map, so it runs even while an orphaned
/// eval holds the sessions write-lock. Threat model: loopback/LAN; the
/// blast radius is "cancel in-flight work", not data access or code
/// execution. Returns how many sessions were signalled.
pub async fn reset_handler(State(state): State<AppState>) -> impl IntoResponse {
    let interrupts = state.interrupts.read().await;
    for entry in interrupts.values() {
        entry.interrupt.set();
    }
    Json(ResetResponse {
        cancelled: interrupts.len(),
    })
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

pub(crate) fn json_err(msg: impl Into<String>) -> Json<ErrorResponse> {
    Json(ErrorResponse { error: msg.into() })
}

/// Saga 21.5 step 003: clone the session's shared `Interrupt`
/// out of the parallel map, `reset()` it (so a prior cancel
/// doesn't contaminate this call), and install it into the
/// session's env. Shared by `eval_handler` (this module) and the
/// SSE `eval_stream_handler` so cancellation behavior is
/// identical across the two transports.
pub(crate) async fn install_session_interrupt(
    state: &AppState,
    id: &Uuid,
    session: &mut crate::sessions::Session,
) {
    if let Some(entry) = state.interrupts.read().await.get(id).cloned() {
        entry.interrupt.reset();
        session.env.set_interrupt(entry.interrupt);
    }
}

pub(crate) fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Array(_) => "array",
        Value::Str(_) => "string",
        Value::Model(_) => "model",
        Value::Tokenizer(_) => "tokenizer",
        Value::DeviceTensor { .. } => "device-tensor",
        Value::BuiltinRef { .. } => "builtin-ref",
        Value::Record { .. } => "record",
        Value::StrList { .. } => "string-list",
        Value::Result { .. } => "result",
    }
}
