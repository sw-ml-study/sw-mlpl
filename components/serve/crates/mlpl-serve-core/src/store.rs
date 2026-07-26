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

use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use mlpl_eval::{EvalError, Value};

use crate::eval_viz::{ErrorResponse, SseEvent, value_kind};

/// Hex-prefix length used in the public URL. 16 hex chars
/// (8 bytes) gives more than enough collision resistance for
/// the loopback / LAN scope while keeping URLs short.
pub const HASH_PREFIX_LEN: usize = 16;

#[derive(Clone)]
pub struct StoredEntry {
    pub bytes: Vec<u8>,
    pub content_type: String,
}

/// In-memory content-addressed store keyed by hex prefix.
#[derive(Default)]
pub struct VizStore {
    pub entries: RwLock<HashMap<String, StoredEntry>>,
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

pub type VizError = (StatusCode, Json<ErrorResponse>);

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
    let hex = content_id(value.as_bytes());
    let entry = StoredEntry {
        bytes: value.as_bytes().to_vec(),
        content_type: "image/svg+xml".into(),
    };
    store.entries.write().await.insert(hex.clone(), entry);
    let local_path = std::env::var("MLPL_CACHE_DIR")
        .ok()
        .and_then(|dir| mlpl_cli::viz_cache::write_to_cache(value, Path::new(&dir)).ok())
        .map(|p| p.display().to_string());
    AttachedViz {
        url: Some(format!("/v1/viz/{hex}")),
        local_path,
    }
}

/// Build the terminal SSE frame from an eval result, attaching
/// a `viz_url` to the `Done` variant when the value is an SVG.
/// Lives here (rather than in lib.rs's `sse` mod) so the inline
/// mod stays under its sw-checklist function-count budget.
pub async fn result_to_sse(store: &SharedVizStore, value: Result<Value, EvalError>) -> SseEvent {
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

/// Content address for stored viz bytes: the first
/// `HASH_PREFIX_LEN` hex chars of the SHA-256. Shared by the
/// eval-pipeline attach path and the `POST /v1/viz` upload
/// handler so identical bytes always mint identical ids.
#[must_use]
pub fn content_id(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
        .chars()
        .take(HASH_PREFIX_LEN)
        .collect()
}
