//! Router wiring + server entry. Saga 21 step 001.

use std::net::SocketAddr;

use axum::Router;
use mlpl_array::DenseArray;
use mlpl_eval::{EvalError, PeerDispatcher, Value};
use serde::{Deserialize, Serialize};

use crate::auth::AuthMode;
use crate::config::{RunConfig, ServeConfig};
use crate::peers::{PeerRegistry, PeerSessionMap};

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

pub use mlpl_serve_state::config::AppState;

#[derive(Debug)]
pub struct RemoteMlxDispatcher {
    peers: PeerRegistry,
    sessions: PeerSessionMap,
}

pub use mlpl_serve_state::peers::EvalOnDeviceBinding;

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
/// `build_app_with_peers_cors` for tests that need to
/// register peers, a CORS origin, or a persistence path
/// up-front.
pub fn build_app(auth_mode: AuthMode) -> Router {
    build_app_with_peers_cors(
        auth_mode,
        crate::peers::empty_registry(),
        ServeConfig::default(),
    )
}

/// Saga R1 step 003 + Saga 25 step A: build the router with an
/// explicit peer registry and an optional static-asset
/// directory. Saga 21.5 step 006 added `cors_origin`: when
/// `Some(o)`, the router is wrapped in a `tower-http`
/// `CorsLayer` that lets a browser on origin `o` reach
/// `/v1/*` (the connect-mode REPL in `apps/mlpl-web`). All
/// four args together replace the old 3-arg
/// `build_app_with_peers`; callers passing `None` for the
/// new arg get the legacy no-CORS behavior.
pub fn build_app_with_peers_cors(
    auth_mode: AuthMode,
    peers: crate::peers::PeerRegistry,
    serve: ServeConfig,
) -> Router {
    let ServeConfig {
        static_dir,
        cors_origin,
        persist_path,
        ollama,
    } = serve;
    let state = AppState::from_parts(auth_mode, peers, persist_path, ollama);
    let router = crate::handlers::v1_router(state);
    mlpl_serve_core::router_layers::apply_static_and_cors(router, static_dir, cors_origin)
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
/// Optional TLS configuration. `None` keeps the legacy
/// HTTP path; `Some(config)` switches to
/// `axum_server::bind_rustls` so the same listener
/// terminates TLS for the /v1 API and the static UI.
pub use mlpl_serve_core::tls::TlsConfig;

pub async fn run(cfg: RunConfig) -> Result<(), ServerError> {
    let RunConfig {
        addr,
        auth_mode,
        peers,
        tls,
        serve,
    } = cfg;
    if !addr.ip().is_loopback() && auth_mode == AuthMode::Disabled {
        return Err(ServerError::InsecureBind { addr });
    }
    let app = build_app_with_peers_cors(auth_mode, peers, serve);
    match tls {
        Some(config) => axum_server::bind_rustls(addr, config)
            .serve(app.into_make_service())
            .await
            .map_err(ServerError::Bind),
        None => {
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(ServerError::Bind)?;
            axum::serve(listener, app).await.map_err(ServerError::Serve)
        }
    }
}
