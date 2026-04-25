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
use mlpl_eval::{Value, eval_program_value};
use mlpl_parser::{lex, parse};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::auth::{AuthMode, check_token, extract_bearer};
use crate::server::AppState;

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
    let value = eval_program_value(&stmts, &mut session.env)
        .map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e}"))))?;
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

fn json_err(msg: impl Into<String>) -> Json<ErrorResponse> {
    Json(ErrorResponse { error: msg.into() })
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Array(_) => "array",
        Value::Str(_) => "string",
        Value::Model(_) => "model",
        Value::Tokenizer(_) => "tokenizer",
    }
}
