//! Router wiring + server entry. Saga R1 step 001.
//!
//! Reuses `mlpl_serve::auth::AuthMode` and the
//! session storage primitives so this server speaks
//! the same auth protocol an orchestrator's
//! `mlpl-serve` does. The route set differs --
//! mlpl-mlx-serve has /health, /sessions, and the
//! step-002-stub /eval-on-device endpoint.

use std::net::SocketAddr;

use axum::Router;
use axum::routing::{get, post};
use mlpl_serve::auth::AuthMode;
use mlpl_serve::sessions::{SessionMap, new_map};

use crate::handlers::{create_session_handler, eval_on_device_stub, health_handler};

/// Errors the server can fail with at startup or
/// while serving. Same shape as
/// `mlpl_serve::server::ServerError` -- duplicated
/// here rather than re-exported because the
/// `Display` text refers to this crate's binary
/// name.
#[derive(Debug)]
pub enum ServerError {
    /// `--bind 0.0.0.0` (or any non-loopback) without
    /// `--auth required`.
    InsecureBind {
        addr: SocketAddr,
    },
    Bind(std::io::Error),
    Serve(std::io::Error),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsecureBind { addr } => write!(
                f,
                "mlpl-mlx-serve refusing to bind {addr} with --auth disabled: \
                 non-loopback addresses require --auth required"
            ),
            Self::Bind(e) => write!(f, "bind failed: {e}"),
            Self::Serve(e) => write!(f, "serve failed: {e}"),
        }
    }
}

impl std::error::Error for ServerError {}

/// Shared state on the application: the session map
/// and the configured auth mode. Mirrors
/// `mlpl_serve::server::AppState` shape.
#[derive(Clone)]
pub struct AppState {
    pub sessions: SessionMap,
    pub auth_mode: AuthMode,
}

/// Build the axum router with the session map state +
/// auth mode wired in. Used by `run` and by the
/// integration tests (which build their own listener
/// on a random port).
pub fn build_app(auth_mode: AuthMode) -> Router {
    let state = AppState {
        sessions: new_map(),
        auth_mode,
    };
    Router::new()
        .route("/v1/health", get(health_handler))
        .route("/v1/sessions", post(create_session_handler))
        .route("/v1/eval-on-device", post(eval_on_device_stub))
        .with_state(state)
}

/// Bind the listener at `addr`, refuse insecure
/// non-loopback binds, then `axum::serve` the
/// router. Used by `main`; tests call `build_app`
/// directly + run on their own listener.
pub async fn run(addr: SocketAddr, auth_mode: AuthMode) -> Result<(), ServerError> {
    if !addr.ip().is_loopback() && auth_mode == AuthMode::Disabled {
        return Err(ServerError::InsecureBind { addr });
    }
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(ServerError::Bind)?;
    axum::serve(listener, build_app(auth_mode))
        .await
        .map_err(ServerError::Serve)
}
