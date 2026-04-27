//! HTTP route handlers. Saga R1 step 002.
//!
//! `/v1/eval-on-device` runs program blocks against
//! a fresh Environment populated with caller-
//! supplied bindings; the result tensor (or string)
//! is stashed on the peer's handle store and
//! returned as an opaque handle. `/transfer` pulls
//! the bytes back. `/release-handle` cleans up.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};
use mlpl_serve::auth::{AuthMode, check_token, extract_bearer};
use mlpl_serve::handlers::{CreateSessionResponse, ErrorResponse};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::server::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub device: &'static str,
}

#[derive(Deserialize)]
pub struct EvalOnDeviceBinding {
    pub name: String,
    pub tensor: String,
}

#[derive(Deserialize)]
pub struct EvalOnDeviceRequest {
    pub program: String,
    #[serde(default)]
    pub bindings: Vec<EvalOnDeviceBinding>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum EvalResultPayload {
    Tensor {
        handle: String,
        shape: Vec<usize>,
        device: &'static str,
    },
    String {
        value: String,
    },
}

#[derive(Serialize)]
pub struct EvalOnDeviceResponse {
    pub result: EvalResultPayload,
}

#[derive(Deserialize)]
pub struct TransferRequest {
    pub handle: String,
}

#[derive(Serialize)]
pub struct TransferResponse {
    pub tensor: String,
}

/// `GET /v1/health` -- no auth.
pub async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        device: "mlx",
    })
}

/// `POST /v1/sessions` -- no auth. Delegates to
/// mlpl-serve's session creation. Step 002 keeps
/// the same shape as step 001; pushing
/// `device("mlx")` onto the env's device stack at
/// creation time stays out -- per-eval push in the
/// eval-on-device handler is the simpler path for
/// now.
pub async fn create_session_handler(State(state): State<AppState>) -> impl IntoResponse {
    let (id, token) = mlpl_serve::sessions::create_session(&state.sessions).await;
    (
        StatusCode::OK,
        Json(CreateSessionResponse {
            session_id: id,
            token,
        }),
    )
}

/// `POST /v1/sessions/{id}/eval-on-device` -- auth
/// required when `auth_mode == Required`. Decodes
/// each binding, runs the program against a fresh
/// sub-Environment, stashes any tensor result on
/// the handle store, returns either a tensor handle
/// or a string value.
pub async fn eval_on_device_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(body): Json<EvalOnDeviceRequest>,
) -> Result<Json<EvalOnDeviceResponse>, (StatusCode, Json<ErrorResponse>)> {
    auth_for_session(&state, id, &headers).await?;
    let mut sub = Environment::new();
    for binding in &body.bindings {
        let arr = crate::wire::decode_from_json(&binding.tensor).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                json_err(format!("binding {:?}: {e}", binding.name)),
            )
        })?;
        sub.set(binding.name.clone(), arr);
    }
    let tokens = lex(&body.program)
        .map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("lex: {e:?}"))))?;
    let stmts =
        parse(&tokens).map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("parse: {e:?}"))))?;
    let value = eval_program_value(&stmts, &mut sub)
        .map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e}"))))?;
    let payload = match value {
        Value::Array(a) => {
            let shape = a.shape().dims().to_vec();
            let handle = crate::handles::insert(&state.handles, a).await;
            EvalResultPayload::Tensor {
                handle: handle.to_string(),
                shape,
                device: "mlx",
            }
        }
        Value::Str(s) => EvalResultPayload::String { value: s },
        Value::Model(_) | Value::Tokenizer(_) | Value::DeviceTensor { .. } => {
            return Err((
                StatusCode::BAD_REQUEST,
                json_err(
                    "eval-on-device blocks must return a tensor or string in R1 \
                     (got model / tokenizer / device-tensor)",
                ),
            ));
        }
    };
    Ok(Json(EvalOnDeviceResponse { result: payload }))
}

/// `POST /v1/sessions/{id}/transfer` -- auth
/// required. Pulls a handle's bytes back via the
/// versioned envelope wire format.
pub async fn transfer_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(body): Json<TransferRequest>,
) -> Result<Json<TransferResponse>, (StatusCode, Json<ErrorResponse>)> {
    auth_for_session(&state, id, &headers).await?;
    let handle_uuid = Uuid::parse_str(&body.handle)
        .map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("handle: {e}"))))?;
    let arr = crate::handles::get(&state.handles, &handle_uuid)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, json_err("unknown handle")))?;
    let tensor = crate::wire::encode_for_json(&arr).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            json_err(format!("encode: {e}")),
        )
    })?;
    Ok(Json(TransferResponse { tensor }))
}

/// `POST /v1/sessions/{id}/release-handle/{handle}`
/// -- auth required. Removes the handle from the
/// store; idempotent (404 if already released).
pub async fn release_handle_handler(
    State(state): State<AppState>,
    Path((id, handle)): Path<(Uuid, String)>,
    headers: HeaderMap,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    auth_for_session(&state, id, &headers).await?;
    let handle_uuid = Uuid::parse_str(&handle)
        .map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("handle: {e}"))))?;
    match crate::handles::release(&state.handles, &handle_uuid).await {
        Some(_) => Ok(StatusCode::NO_CONTENT),
        None => Err((StatusCode::NOT_FOUND, json_err("unknown handle"))),
    }
}

async fn auth_for_session(
    state: &AppState,
    id: Uuid,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let sessions = state.sessions.read().await;
    let session = sessions
        .get(&id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, json_err("unknown session")))?;
    if state.auth_mode == AuthMode::Required {
        let provided = extract_bearer(headers).ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                json_err("missing or invalid authorization"),
            )
        })?;
        if !check_token(provided, &session.token) {
            return Err((
                StatusCode::UNAUTHORIZED,
                json_err("missing or invalid authorization"),
            ));
        }
    }
    Ok(())
}

fn json_err(msg: impl Into<String>) -> Json<ErrorResponse> {
    Json(ErrorResponse { error: msg.into() })
}
