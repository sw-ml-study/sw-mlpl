//! Optional router layers applied after the core `/v1` routes: the
//! static-asset mount (`/sw-mlpl`) and the connect-mode CORS layer.
//! Extracted from `build_app_with_peers_cors` to keep that builder
//! under the sw-checklist function-LOC budget.

use std::path::PathBuf;

use axum::Router;

/// Apply the optional static-dir nest + CORS layer to `router`.
/// `static_dir` mounts the UI under `/sw-mlpl`; `cors_origin` wraps
/// `/v1/*` in a `tower-http` CORS layer for a cross-origin browser
/// REPL. Each is a no-op when `None`.
pub fn apply_static_and_cors(
    mut router: Router,
    static_dir: Option<PathBuf>,
    cors_origin: Option<String>,
) -> Router {
    if let Some(dir) = static_dir.as_deref() {
        let serve = tower_http::services::ServeDir::new(dir);
        router = router.nest_service("/sw-mlpl", serve);
    }
    if let Some(origin) = cors_origin {
        use axum::http::{Method, header};
        let origin_header = origin
            .parse::<axum::http::HeaderValue>()
            .expect("--cors-allow value must be a valid origin header");
        let layer = tower_http::cors::CorsLayer::new()
            .allow_origin(origin_header)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]);
        router = router.layer(layer);
    }
    router
}
