//! HTTP handlers for the viz byte store (`POST /v1/viz`,
//! `GET /v1/viz/:id`). The store itself lives in mlpl-serve-core.

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Json, Response};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;

use mlpl_serve_core::auth::{AuthMode, check_token, extract_bearer};
use mlpl_serve_core::eval_viz::json_err;
use mlpl_serve_core::store::*;

use crate::config::AppState;

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
    let hex = mlpl_serve_core::store::content_id(&bytes);
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
