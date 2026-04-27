//! Router wiring + server entry. Saga 21 step 001.

use std::net::SocketAddr;

use axum::Router;
use axum::routing::{get, post};
use mlpl_array::DenseArray;
use mlpl_eval::{EvalError, PeerDispatcher, Value};
use serde::{Deserialize, Serialize};

use crate::auth::AuthMode;
use crate::handlers::{create_session_handler, eval_handler, health_handler, inspect_handler};
use crate::peers::{PeerRegistry, PeerSessionMap};
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
    pub peers: PeerRegistry,
    pub peer_sessions: PeerSessionMap,
    pub auth_mode: AuthMode,
}

#[derive(Debug)]
pub struct RemoteMlxDispatcher {
    peers: PeerRegistry,
    sessions: PeerSessionMap,
}

#[derive(Serialize)]
pub struct EvalOnDeviceBinding {
    pub name: String,
    pub tensor: String,
}

#[derive(Serialize)]
struct EvalOnDeviceRequest {
    program: String,
    bindings: Vec<EvalOnDeviceBinding>,
}

#[derive(Deserialize)]
struct EvalOnDeviceResponse {
    result: EvalResultPayload,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum EvalResultPayload {
    Tensor {
        handle: String,
        shape: Vec<usize>,
        device: String,
    },
    String {
        value: String,
    },
}

#[derive(Deserialize)]
struct TransferResponse {
    tensor: String,
}

impl RemoteMlxDispatcher {
    #[must_use]
    pub fn new(peers: PeerRegistry, sessions: PeerSessionMap) -> Self {
        Self { peers, sessions }
    }
}

impl PeerDispatcher for RemoteMlxDispatcher {
    #[rustfmt::skip]
    fn dispatch_block(
        &self,
        device: &str,
        source: &str,
        bindings: std::collections::HashMap<String, DenseArray>,
    ) -> Option<Result<Value, EvalError>> {
        let peer = self.peers.get(device)?.clone();
        let session = self.sessions.get_or_create(&peer);
        let bindings = crate::peers::encode_bindings(bindings);
        let (session, bindings) = match (session, bindings) {
            (Ok(session), Ok(bindings)) => (session, bindings),
            (Err(e), _) | (_, Err(e)) => return Some(Err(e)),
        };
        let body = EvalOnDeviceRequest { program: source.to_string(), bindings };
        let url = format!(
            "{}/v1/sessions/{}/eval-on-device",
            peer.url.trim_end_matches('/'),
            session.id
        );
        let client = peer.client;
        let token = session.token;
        let result = std::thread::spawn(move || {
            client
                .post(url)
                .bearer_auth(&token)
                .json(&body)
                .send()
                .and_then(reqwest::blocking::Response::error_for_status)?
                .json::<EvalOnDeviceResponse>()
        })
        .join()
        .map_err(|_| EvalError::Unsupported("remote peer thread panicked".into()));
        Some(result.and_then(|r| {
            r.map_err(|e| EvalError::Unsupported(format!("remote peer request: {e}")))
        }).map(|r| match r.result {
            EvalResultPayload::Tensor { handle, shape, device } => {
                Value::DeviceTensor { peer: peer.url, handle, shape, device }
            }
            EvalResultPayload::String { value } => Value::Str(value),
        }))
    }

    fn fetch_tensor(&self, peer_url: &str, handle: &str) -> Result<DenseArray, EvalError> {
        let peer = self
            .peers
            .values()
            .find(|p| p.url == peer_url)
            .ok_or_else(|| EvalError::Unsupported(format!("unknown peer {peer_url}")))?;
        let session = self.sessions.get_or_create(peer)?;
        let url = format!(
            "{}/v1/sessions/{}/transfer",
            peer.url.trim_end_matches('/'),
            session.id
        );
        let client = peer.client;
        let token = session.token;
        let handle = handle.to_string();
        let resp = std::thread::spawn(move || {
            client
                .post(url)
                .bearer_auth(&token)
                .json(&serde_json::json!({ "handle": handle }))
                .send()
                .and_then(reqwest::blocking::Response::error_for_status)?
                .json::<TransferResponse>()
        })
        .join()
        .map_err(|_| EvalError::Unsupported("remote peer thread panicked".into()))?
        .map_err(|e| EvalError::Unsupported(format!("remote peer request: {e}")))?;
        crate::peers::decode_from_json(&resp.tensor)
    }
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
        peer_sessions: PeerSessionMap::default(),
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
