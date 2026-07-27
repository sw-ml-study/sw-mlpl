//! Optional router layers applied after the core `/v1` routes: the
//! static-asset mount (`/sw-mlpl`) and the connect-mode CORS layer.
//! Extracted from `build_app_with_peers_cors` to keep that builder
//! under the sw-checklist function-LOC budget.

use std::path::PathBuf;

use axum::Router;

/// Parse a `--cors-allow` value: one origin, or a comma-separated
/// list (a server often fronts several page hosts -- a trunk dev
/// page and a static host). Entries are trimmed; empty entries are
/// dropped; an unparsable origin panics naming the flag.
#[must_use]
pub fn parse_cors_origins(spec: &str) -> Vec<axum::http::HeaderValue> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .expect("--cors-allow value must be a valid origin header")
        })
        .collect()
}

/// Apply the optional static-dir nest + CORS layer to `router`.
/// `static_dir` mounts the UI under `/sw-mlpl`; `cors_origin` wraps
/// `/v1/*` in a `tower-http` CORS layer for cross-origin browser
/// REPLs (one origin or a comma-separated list). Each is a no-op
/// when `None`.
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
        let layer = tower_http::cors::CorsLayer::new()
            .allow_origin(tower_http::cors::AllowOrigin::list(parse_cors_origins(
                &origin,
            )))
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]);
        router = router.layer(layer);
    }
    router
}
