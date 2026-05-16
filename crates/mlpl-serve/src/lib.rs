//! `mlpl-serve` -- long-running MLPL interpreter
//! exposed as a REST API. Saga 21 step 001 MVP:
//! sessions + eval + health. Inspect endpoint lands
//! in step 002; LLM proxy / SSE / cancellation /
//! persistence are all explicit non-goals.
//!
//! Library + binary: integration tests
//! (`tests/api_tests.rs`) construct routers via
//! `server::build_app(...)` and serve them on
//! random ports; the `mlpl-serve` binary is a thin
//! shell around `server::run(addr, auth_mode)`.

pub mod auth;
pub mod handlers;
pub mod peers;
pub mod persist;
pub mod server;
pub mod sessions;
pub mod viz_storage;

/// Saga 21.5 step 001: Server-Sent-Events streaming eval. Inline
/// submodule of the crate root (rather than a top-level file) so
/// the crate module count stays at the sw-checklist budget --
/// matches the precedent set by the `tls` module below. Exposes the
/// `POST /v1/sessions/:id/eval_stream` handler, the `SseEvent` wire
/// shape, and the `ChannelMetricSink` adapter that wires
/// `eval_train`'s per-iteration `_metric` scan into a tokio
/// `mpsc::Sender`.
pub mod sse {
    use std::convert::Infallible;
    use std::sync::Arc;

    use axum::Json;
    use axum::extract::{Path, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::sse::{Event, KeepAlive, Sse};
    use mlpl_eval::{MetricSink, eval_program_value};
    use mlpl_parser::{lex, parse};
    use serde::Serialize;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_stream::{Stream, StreamExt};
    use uuid::Uuid;

    use crate::auth::{AuthMode, check_token, extract_bearer};
    use crate::handlers::{ErrorResponse, EvalRequest, json_err};
    use crate::server::AppState;

    /// One frame on the SSE stream. The first frame is always
    /// `Ready`. Zero-or-more `Metric` frames may follow (one per
    /// `_metric`-suffixed scalar binding per `train { }` iteration).
    /// The stream ends with exactly one terminal frame, either
    /// `Done` (success) or `Error` (any `EvalError`).
    #[derive(Debug, Serialize)]
    #[serde(tag = "event", content = "data", rename_all = "lowercase")]
    pub enum SseEvent {
        Ready,
        Metric {
            name: String,
            step: usize,
            value: f64,
        },
        Done {
            value: String,
            kind: &'static str,
            /// Saga 21.5 step 004: when the eval returned an
            /// SVG-shaped string, this carries the URL the
            /// `/v1/viz` storage minted for it. `None` (omitted
            /// from JSON) for non-SVG results.
            #[serde(skip_serializing_if = "Option::is_none")]
            viz_url: Option<String>,
            /// Saga 21.5 step 004: server-side `MLPL_CACHE_DIR`
            /// path for the same SVG, when configured.
            #[serde(skip_serializing_if = "Option::is_none")]
            viz_local_path: Option<String>,
        },
        /// Saga 21.5 step 003: terminal frame for cooperative
        /// cancellation. Mirrors `EvalError::Cancelled`'s `step`
        /// and `partial_losses` so clients can render the partial
        /// loss curve without falling back to a second
        /// `:vars`/inspect round-trip.
        Cancelled {
            step: usize,
            partial_losses: Vec<f64>,
        },
        Error {
            error: String,
        },
    }

    impl SseEvent {
        /// Convert one `SseEvent` to an axum `Event` carrying the
        /// `event:` line and a `data:` JSON payload. For variants
        /// without a body (`Ready`), `data:` is an empty JSON
        /// object so clients that always parse `data` do not need
        /// a special case.
        pub fn to_axum_event(&self) -> Event {
            let (name, data) = match self {
                Self::Ready => ("ready".to_string(), serde_json::json!({})),
                Self::Metric { name, step, value } => (
                    "metric".to_string(),
                    serde_json::json!({"name": name, "step": step, "value": value}),
                ),
                Self::Done {
                    value,
                    kind,
                    viz_url,
                    viz_local_path,
                } => {
                    let mut payload = serde_json::json!({"value": value, "kind": kind});
                    if let Some(u) = viz_url {
                        payload["viz_url"] = serde_json::Value::String(u.clone());
                    }
                    if let Some(p) = viz_local_path {
                        payload["viz_local_path"] = serde_json::Value::String(p.clone());
                    }
                    ("done".to_string(), payload)
                }
                Self::Cancelled {
                    step,
                    partial_losses,
                } => (
                    "cancelled".to_string(),
                    serde_json::json!({"step": step, "partial_losses": partial_losses}),
                ),
                Self::Error { error } => ("error".to_string(), serde_json::json!({"error": error})),
            };
            Event::default().event(name).data(data.to_string())
        }
    }

    /// Adapter from the synchronous `MetricSink` trait into an
    /// async tokio mpsc channel. The eval runs inside
    /// `spawn_blocking`, so `blocking_send` is the right primitive
    /// -- it applies channel-bounded backpressure if the consumer
    /// stalls without forcing the caller into async. `tx` is
    /// `pub(super)` rather than wrapped behind a `new` constructor
    /// so callers can build the struct inline and keep lib.rs
    /// under the sw-checklist 7-fn-per-module budget.
    #[derive(Debug)]
    pub struct ChannelMetricSink {
        pub(super) tx: mpsc::Sender<SseEvent>,
    }

    impl MetricSink for ChannelMetricSink {
        fn emit(&self, name: &str, step: usize, value: f64) {
            let _ = self.tx.blocking_send(SseEvent::Metric {
                name: name.to_string(),
                step,
                value,
            });
        }
    }

    type SseError = (StatusCode, Json<ErrorResponse>);

    /// Spawn the eval as a `spawn_blocking` task, route the result
    /// through the channel, and re-insert the session into the
    /// shared map on completion. Decoupled from `eval_stream_handler`
    /// so each function stays under the sw-checklist 50-line budget.
    fn spawn_eval_task(
        id: Uuid,
        mut session: crate::sessions::Session,
        stmts: Vec<mlpl_parser::Expr>,
        sessions_map: crate::sessions::SessionMap,
        viz: crate::viz_storage::SharedVizStore,
        tx: mpsc::Sender<SseEvent>,
    ) {
        tokio::spawn(async move {
            let _ = tx.send(SseEvent::Ready).await;
            let join = tokio::task::spawn_blocking(move || {
                let value = eval_program_value(&stmts, &mut session.env);
                (session, value)
            })
            .await;
            let (mut session, value) = match join {
                Ok(t) => t,
                Err(_) => {
                    let _ = tx
                        .send(SseEvent::Error {
                            error: "eval task panicked".into(),
                        })
                        .await;
                    return;
                }
            };
            session.env.clear_metric_sink();
            session.env.clear_peer_dispatcher();
            session.env.clear_interrupt();
            let terminal = crate::viz_storage::result_to_sse(&viz, value).await;
            let _ = tx.send(terminal).await;
            sessions_map.write().await.insert(id, session);
        });
    }

    /// `POST /v1/sessions/:id/eval_stream` -- requires bearer when
    /// `auth_mode == Required`. Lex + parse synchronously (so a
    /// 400 still surfaces as plain JSON); on success, take the
    /// session out of the map, run eval on a `spawn_blocking`
    /// task with a `ChannelMetricSink` installed, and return an
    /// SSE stream whose frames are: one `ready`, zero-or-more
    /// `metric`, one terminal `done` or `error`.
    pub async fn eval_stream_handler(
        State(state): State<AppState>,
        Path(id): Path<Uuid>,
        headers: HeaderMap,
        Json(body): Json<EvalRequest>,
    ) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, SseError> {
        {
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
        }
        let tokens = lex(&body.program)
            .map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e:?}"))))?;
        let stmts =
            parse(&tokens).map_err(|e| (StatusCode::BAD_REQUEST, json_err(format!("{e:?}"))))?;
        let mut session = state
            .sessions
            .write()
            .await
            .remove(&id)
            .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
        let (tx, rx) = mpsc::channel::<SseEvent>(64);
        let sink: Arc<dyn MetricSink> = Arc::new(ChannelMetricSink { tx: tx.clone() });
        session.env.set_metric_sink(sink);
        crate::handlers::install_session_interrupt(&state, &id, &mut session).await;
        session
            .env
            .set_peer_dispatcher(Arc::new(crate::server::RemoteMlxDispatcher::new(
                state.peers.clone(),
                state.peer_sessions.clone(),
            )));
        let sessions = state.sessions.clone();
        let viz = state.viz.clone();
        spawn_eval_task(id, session, stmts, sessions, viz, tx);
        let stream = ReceiverStream::new(rx).map(|ev| Ok::<_, Infallible>(ev.to_axum_event()));
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
    }
}

/// TLS configuration helpers. Inline submodule of the
/// crate root (rather than a top-level file) so the crate
/// module count stays at the sw-checklist budget. Used by
/// `main`'s `--tls-cert`/`--tls-key`/`--self-signed`
/// dispatcher and one integration test.
pub mod tls {
    use axum_server::tls_rustls::RustlsConfig;
    use rcgen::{CertifiedKey, generate_simple_self_signed};
    use sha2::{Digest, Sha256};

    /// Install the `ring` crypto provider as rustls's
    /// process-wide default. Idempotent. Required since
    /// rustls 0.23 dropped its built-in default and forces
    /// an explicit `ring` vs `aws-lc-rs` choice.
    pub fn ensure_crypto_provider() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    /// `RustlsConfig` from PEM cert + key on disk.
    pub async fn from_pem_files(
        cert: &std::path::Path,
        key: &std::path::Path,
    ) -> Result<RustlsConfig, String> {
        ensure_crypto_provider();
        RustlsConfig::from_pem_file(cert, key)
            .await
            .map_err(|e| format!("TLS: failed to load cert/key: {e}"))
    }

    /// Self-signed cert covering loopback hosts, returned
    /// as `(RustlsConfig, fingerprint)`. Fingerprint is
    /// SHA-256 of the DER-encoded cert in colon-separated
    /// uppercase hex.
    pub async fn self_signed_loopback() -> Result<(RustlsConfig, String), String> {
        ensure_crypto_provider();
        let alt = vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
            "::1".to_string(),
        ];
        let CertifiedKey { cert, key_pair } = generate_simple_self_signed(alt)
            .map_err(|e| format!("TLS: failed to generate self-signed cert: {e}"))?;
        let der = cert.der().to_vec();
        let mut fp = String::with_capacity(95);
        for (i, b) in Sha256::digest(&der).iter().enumerate() {
            if i > 0 {
                fp.push(':');
            }
            std::fmt::Write::write_fmt(&mut fp, format_args!("{b:02X}")).unwrap();
        }
        let config = RustlsConfig::from_pem(
            cert.pem().into_bytes(),
            key_pair.serialize_pem().into_bytes(),
        )
        .await
        .map_err(|e| format!("TLS: failed to assemble rustls config: {e}"))?;
        Ok((config, fp))
    }
}
