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
use mlpl_eval::Interrupt;
use mlpl_parser::{lex, parse};
use serde::Serialize;
use uuid::Uuid;

use crate::server::AppState;

// Wire/response types + helpers shared with the lower layers live in
// mlpl-serve-core; re-exported so crate::handlers:: paths keep working.
pub use mlpl_serve_core::eval_viz::{
    ErrorResponse, EvalRequest, EvalResponse, json_err, value_kind,
};

// Connect-telemetry step 003 (ratchet): the introspection handlers and
// the eval-task machinery moved to sibling modules; re-export so
// existing `crate::handlers::` / `mlpl_serve::handlers::` paths hold.
pub(crate) use crate::handlers_eval_task::{spawn_eval, take_session_for_eval};
pub use mlpl_serve_state::handlers_inspect::{
    InspectResponse, SessionMetaResponse, VarSnapshot, inspect_handler, session_meta_handler,
};

#[derive(Serialize)]
pub struct CreateSessionResponse {
    pub session_id: Uuid,
    pub token: String,
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
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
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

/// `POST /v1/sessions/{id}/eval` -- requires bearer when
/// `auth_mode == Required`. Lex + parse, then run the program OFF the
/// global session lock: the session is taken out of the map, evaluated
/// on a `spawn_blocking` task, and reinserted -- so a long eval never
/// blocks new-session creation or other serving (notably the telemetry
/// panel's `/v1/stats`). An `AbortGuard` trips the session interrupt if
/// the client disconnects mid-eval (a shift-reload), so the eval aborts
/// at its next checkpoint instead of orphaning and wedging the server.
pub async fn eval_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(body): Json<EvalRequest>,
) -> Result<Json<EvalResponse>, (StatusCode, Json<ErrorResponse>)> {
    let stmts = parse_program(&body.program)?;
    let (session, interrupt) = take_session_for_eval(&state, id, &headers).await?;
    let rx = spawn_eval(&state, id, session, stmts, body.program.clone());
    let mut guard = AbortGuard {
        interrupt,
        armed: true,
    };
    let result = rx.await;
    guard.armed = false;
    match result {
        Ok(Ok(resp)) => Ok(Json(resp)),
        Ok(Err(e)) => Err((StatusCode::BAD_REQUEST, json_err(e))),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            json_err("eval task dropped"),
        )),
    }
}

/// Trips a session's interrupt when dropped while still armed -- the
/// cancel-on-disconnect mechanism. `eval_handler` disarms it on normal
/// completion; if the handler future is instead dropped (client
/// disconnect), the drop sets the interrupt and the in-flight eval
/// aborts at its next checkpoint.
struct AbortGuard {
    interrupt: Interrupt,
    armed: bool,
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        if self.armed {
            self.interrupt.set();
        }
    }
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
    let entry = state
        .interrupts
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
    mlpl_serve_core::sessions::require_bearer(state.auth_mode, &entry.token, &headers)?;
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

/// Lex + parse `program`, mapping either failure to a 400 with the
/// debug-formatted error. Shared by `/eval` and `/eval_stream`.
pub(crate) fn parse_program(
    program: &str,
) -> Result<Vec<mlpl_parser::Expr>, (StatusCode, Json<ErrorResponse>)> {
    let tokens = lex(program).map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e:?}"))))?;
    parse(&tokens).map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e:?}"))))
}

/// The full `/v1` route table over a ready `AppState`. Extracted
/// from `server::build_app_with_peers_cors` for the LOC budget.
pub(crate) fn v1_router(state: AppState) -> axum::Router {
    use axum::routing::{get, post};
    axum::Router::new()
        .route("/v1/health", get(health_handler))
        .route(
            "/v1/devices",
            get(mlpl_serve_core::devices::devices_handler),
        )
        .route("/v1/stats", get(mlpl_serve_core::devices::stats_handler))
        .route("/v1/reset", post(reset_handler))
        .route("/v1/sessions", post(create_session_handler))
        .route("/v1/sessions/:id", get(session_meta_handler))
        .route("/v1/sessions/:id/eval", post(eval_handler))
        .route(
            "/v1/sessions/:id/eval_stream",
            post(crate::sse::eval_stream_handler),
        )
        .route("/v1/sessions/:id/cancel", post(cancel_handler))
        .route("/v1/sessions/:id/inspect", get(inspect_handler))
        .route(
            "/v1/viz",
            post(mlpl_serve_state::viz_handlers::upload_handler),
        )
        .route(
            "/v1/viz/:id",
            get(mlpl_serve_state::viz_handlers::get_handler),
        )
        .route(
            "/v1/ollama/config",
            get(mlpl_serve_state::ollama::config_handler),
        )
        .route(
            "/v1/ollama/tags",
            get(mlpl_serve_state::ollama::tags_handler),
        )
        .with_state(state)
}
