//! HTTP route handlers. Saga R1 step 001.
//!
//! `/eval-on-device` returns 501 in this step; real
//! implementation ships in step 002 once the wire
//! format module lands. `/sessions` reuses
//! `mlpl_serve::sessions::create_session` for now;
//! step 002 will wrap that to push `device("mlx")`
//! onto the env's device stack at session creation.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use mlpl_serve::handlers::{CreateSessionResponse, ErrorResponse};
use serde::Serialize;
use uuid::Uuid;

use crate::server::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub device: &'static str,
}

/// `GET /v1/health` -- no auth. Reports the
/// device this server manages so an orchestrator
/// can confirm it registered the right peer URL.
pub async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        device: "mlx",
    })
}

/// `POST /v1/sessions` -- no auth (this is how
/// clients get their bearer token). Delegates to
/// `mlpl_serve::sessions::create_session`. Step
/// 002 will wrap this to push `device("mlx")` onto
/// the env's device stack at session creation
/// time.
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

/// `POST /v1/eval-on-device` -- STUB. Real
/// implementation lands in Saga R1 step 002 with
/// the bincode + versioned-envelope tensor wire
/// format. Returns 501 Not Implemented with an
/// actionable forward-pointer.
pub async fn eval_on_device_stub(
    State(_state): State<AppState>,
    _body: Option<Json<serde_json::Value>>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let _: Uuid = Uuid::nil();
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "wire format ships in Saga R1 step 002; \
                    POST /v1/eval-on-device is a stub today"
                .into(),
        }),
    ))
}
