//! Saga 21.5 step 004: content-addressed viz storage.
//!
//! `POST /v1/viz` accepts a base64 payload + content_type and
//! returns `/v1/viz/<sha256-hex-prefix>`. `GET /v1/viz/:id`
//! serves the bytes back with the recorded content type. The
//! eval pipeline (`eval_handler` and the SSE `spawn_eval_task`)
//! detect SVG-returning programs and stash the bytes here so the
//! non-streaming and streaming responses both carry a `viz_url`.
//!
//! Top-level module file (rather than an inline submodule of
//! `lib.rs`) because the inline form pushes lib.rs over its
//! sw-checklist function-count + file-LOC budget. The
//! `Crate Module Count` warning ticks up by one in exchange.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Json, Response};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use mlpl_eval::{EvalError, Value};

use crate::auth::{AuthMode, check_token, extract_bearer};
use crate::handlers::{ErrorResponse, json_err, value_kind};
use crate::server::AppState;
use crate::sse::SseEvent;

/// Hex-prefix length used in the public URL. 16 hex chars
/// (8 bytes) gives more than enough collision resistance for
/// the loopback / LAN scope while keeping URLs short.
pub(crate) const HASH_PREFIX_LEN: usize = 16;

#[derive(Clone)]
struct StoredEntry {
    bytes: Vec<u8>,
    content_type: String,
}

/// In-memory content-addressed store keyed by hex prefix.
#[derive(Default)]
pub struct VizStore {
    entries: RwLock<HashMap<String, StoredEntry>>,
}

/// Shared handle to the viz store, embedded in `AppState`.
pub type SharedVizStore = Arc<VizStore>;

/// Construct a fresh empty store handle.
#[must_use]
pub fn new_store() -> SharedVizStore {
    Arc::new(VizStore::default())
}

#[derive(Deserialize)]
pub struct UploadRequest {
    pub bytes_base64: String,
    pub content_type: String,
}

#[derive(Serialize)]
pub struct UploadResponse {
    pub id: String,
    pub url: String,
}

type VizError = (StatusCode, Json<ErrorResponse>);

/// Saga 21.5 step 004: per-eval viz attachment.
///
/// `Empty` for non-string / non-SVG values. `Stored` carries
/// the `viz_url` (always) and the `viz_local_path` (when the
/// server has `MLPL_CACHE_DIR` set and the write succeeded).
/// Used by both `handlers::eval_handler` and the SSE
/// `spawn_eval_task`.
#[derive(Debug, Default)]
pub struct AttachedViz {
    pub url: Option<String>,
    pub local_path: Option<String>,
}

/// Detect an SVG-returning eval and stash the bytes in the
/// store + the server-side `MLPL_CACHE_DIR` (when set).
pub async fn attach_viz(store: &SharedVizStore, value: &str, kind: &str) -> AttachedViz {
    if kind != "string" || !mlpl_cli::viz_cache::is_svg_string(value) {
        return AttachedViz::default();
    }
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let hex: String = format!("{:x}", hasher.finalize())
        .chars()
        .take(HASH_PREFIX_LEN)
        .collect();
    store.entries.write().await.insert(
        hex.clone(),
        StoredEntry {
            bytes: value.as_bytes().to_vec(),
            content_type: "image/svg+xml".into(),
        },
    );
    let url = format!("/v1/viz/{hex}");
    let local_path = std::env::var("MLPL_CACHE_DIR")
        .ok()
        .and_then(|dir| mlpl_cli::viz_cache::write_to_cache(value, Path::new(&dir)).ok())
        .map(|p| p.display().to_string());
    AttachedViz {
        url: Some(url),
        local_path,
    }
}

/// Build the terminal SSE frame from an eval result, attaching
/// a `viz_url` to the `Done` variant when the value is an SVG.
/// Lives here (rather than in lib.rs's `sse` mod) so the inline
/// mod stays under its sw-checklist function-count budget.
pub(crate) async fn result_to_sse(
    store: &SharedVizStore,
    value: Result<Value, EvalError>,
) -> SseEvent {
    match value {
        Ok(v) => {
            let formatted = format!("{v}");
            let kind = value_kind(&v);
            let a = attach_viz(store, &formatted, kind).await;
            SseEvent::Done {
                value: formatted,
                kind,
                viz_url: a.url,
                viz_local_path: a.local_path,
            }
        }
        Err(EvalError::Cancelled {
            step,
            partial_losses,
        }) => SseEvent::Cancelled {
            step,
            partial_losses,
        },
        Err(e) => SseEvent::Error {
            error: format!("{e}"),
        },
    }
}

/// `POST /v1/viz` -- requires bearer when `auth_mode ==
/// Required`. Accepts `{bytes_base64, content_type}` and returns
/// `{id, url}`. Idempotent: identical bytes yield identical ids.
pub async fn upload_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<UploadRequest>,
) -> Result<Json<UploadResponse>, VizError> {
    require_known_bearer(&headers, &state).await?;
    let bytes = BASE64.decode(body.bytes_base64.as_bytes()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            json_err(format!("base64 decode: {e}")),
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hex: String = format!("{:x}", hasher.finalize())
        .chars()
        .take(HASH_PREFIX_LEN)
        .collect();
    state.viz.entries.write().await.insert(
        hex.clone(),
        StoredEntry {
            bytes,
            content_type: body.content_type,
        },
    );
    let url = format!("/v1/viz/{hex}");
    Ok(Json(UploadResponse { id: hex, url }))
}

/// `GET /v1/viz/:id` -- requires bearer when `auth_mode ==
/// Required`. Returns the stored bytes with the recorded
/// `Content-Type`. 404 on unknown id.
pub async fn get_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
) -> Result<Response, VizError> {
    require_known_bearer(&headers, &state).await?;
    let entry = state
        .viz
        .entries
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown viz id")))?;
    let mut resp = Response::new(Body::from(entry.bytes));
    if let Ok(value) = HeaderValue::from_str(&entry.content_type) {
        resp.headers_mut().insert(header::CONTENT_TYPE, value);
    }
    Ok(resp)
}

/// Shared auth check for the two `/v1/viz` handlers. The bearer
/// must match SOME existing session's token (looked up against
/// the parallel `InterruptMap` so it does NOT block on the
/// sessions write lock held by an in-flight eval). No-op when
/// `auth_mode == Disabled`.
async fn require_known_bearer(headers: &HeaderMap, state: &AppState) -> Result<(), VizError> {
    if state.auth_mode != AuthMode::Required {
        return Ok(());
    }
    let provided = extract_bearer(headers).ok_or((
        StatusCode::UNAUTHORIZED,
        json_err("missing or invalid authorization"),
    ))?;
    let interrupts = state.interrupts.read().await;
    if interrupts.values().any(|e| check_token(provided, &e.token)) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            json_err("missing or invalid authorization"),
        ))
    }
}
