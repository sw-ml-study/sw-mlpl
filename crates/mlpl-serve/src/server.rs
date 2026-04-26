//! Router wiring + server entry. Saga 21 step 001.

use std::net::SocketAddr;

use axum::Router;
use axum::routing::{get, post};

use crate::auth::AuthMode;
use crate::handlers::{create_session_handler, eval_handler, health_handler, inspect_handler};
use crate::sessions::{SessionMap, new_map};

/// Errors the server can fail with at startup or
/// while serving. Translated to stderr + non-zero
/// exit by `main`.
#[derive(Debug)]
pub enum ServerError {
    /// `--bind 0.0.0.0` (or any non-loopback) without
    /// `--auth required`.
    InsecureBind { addr: SocketAddr },
    /// Failed to bind the listener socket.
    Bind(std::io::Error),
    /// Axum `serve` returned an error.
    Serve(std::io::Error),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsecureBind { addr } => write!(
                f,
                "refusing to bind {addr} with --auth disabled: \
                 non-loopback addresses require --auth required"
            ),
            Self::Bind(e) => write!(f, "bind failed: {e}"),
            Self::Serve(e) => write!(f, "serve failed: {e}"),
        }
    }
}

impl std::error::Error for ServerError {}

/// Shared state on the application: the session map,
/// the peer registry (Saga R1 step 003), and the
/// configured auth mode.
#[derive(Clone)]
pub struct AppState {
    pub sessions: SessionMap,
    pub peers: crate::peers::PeerRegistry,
    pub auth_mode: AuthMode,
}

/// Build the axum router with the session-map state
/// and auth mode wired in. Empty peer registry; use
/// `build_app_with_peers` for tests that need to
/// register peers up-front.
pub fn build_app(auth_mode: AuthMode) -> Router {
    build_app_with_peers(auth_mode, crate::peers::empty_registry())
}

/// Saga R1 step 003: build the router with an
/// explicit peer registry. Used by `run` and by the
/// peer-routing integration test harness.
pub fn build_app_with_peers(auth_mode: AuthMode, peers: crate::peers::PeerRegistry) -> Router {
    let state = AppState {
        sessions: new_map(),
        peers,
        auth_mode,
    };
    Router::new()
        .route("/v1/health", get(health_handler))
        .route("/v1/sessions", post(create_session_handler))
        .route("/v1/sessions/:id/eval", post(eval_handler))
        .route("/v1/sessions/:id/inspect", get(inspect_handler))
        .with_state(state)
}

/// Bind the listener at `addr`, refuse insecure
/// non-loopback binds, then `axum::serve` the
/// router. Used by `main`; tests call `build_app`
/// directly + run on their own listener.
///
/// Saga R1 step 003: `peers` carries the
/// `--peer mlx=<url>`-derived registry built up by
/// `peers::build_registry` in `main`. An empty
/// registry means "no peer routing; device-scoped
/// blocks fall back to in-process dispatch."
pub async fn run(
    addr: SocketAddr,
    auth_mode: AuthMode,
    peers: crate::peers::PeerRegistry,
) -> Result<(), ServerError> {
    if !addr.ip().is_loopback() && auth_mode == AuthMode::Disabled {
        return Err(ServerError::InsecureBind { addr });
    }
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(ServerError::Bind)?;
    axum::serve(listener, build_app_with_peers(auth_mode, peers))
        .await
        .map_err(ServerError::Serve)
}
