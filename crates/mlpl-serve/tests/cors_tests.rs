//! Saga 21.5 step 006: `--cors-allow <origin>` flag tests.
//!
//! `mlpl-web` running on `https://sw-ml-study.github.io/sw-mlpl/`
//! cannot fetch a `mlpl-serve` instance on a different origin
//! (a loopback dev server, say) unless the server sends
//! `Access-Control-Allow-Origin`. The new `--cors-allow` flag
//! threads an origin string into `build_app_with_peers_cors`,
//! which wraps the router in a `tower-http` `CorsLayer`.

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::peers::empty_registry;
use mlpl_serve::server::build_app_with_peers_cors;

async fn start_server_with_cors(origin: Option<&str>) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app_with_peers_cors(
        AuthMode::Required,
        empty_registry(),
        None,
        origin.map(str::to_string),
    );
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

/// Health-check response carries the `Access-Control-Allow-Origin`
/// header when `--cors-allow <origin>` matches the request's
/// `Origin`.
#[tokio::test(flavor = "multi_thread")]
async fn cors_allow_adds_allow_origin_header_to_responses() {
    let addr = start_server_with_cors(Some("https://example.test")).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/health"))
        .header("Origin", "https://example.test")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let value = resp
        .headers()
        .get("access-control-allow-origin")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_default();
    assert_eq!(value, "https://example.test");
}

/// Preflight `OPTIONS` request with `Access-Control-Request-*`
/// headers succeeds when the origin is in the allow-list. The
/// response advertises the bearer-token `Authorization` header
/// as allowed so the browser will let the actual POST through.
#[tokio::test(flavor = "multi_thread")]
async fn cors_preflight_for_eval_succeeds() {
    let addr = start_server_with_cors(Some("https://example.test")).await;
    let resp = reqwest::Client::new()
        .request(
            reqwest::Method::OPTIONS,
            format!("http://{addr}/v1/sessions"),
        )
        .header("Origin", "https://example.test")
        .header("Access-Control-Request-Method", "POST")
        .header(
            "Access-Control-Request-Headers",
            "authorization,content-type",
        )
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "preflight should succeed, got {}",
        resp.status()
    );
    let allow_origin = resp
        .headers()
        .get("access-control-allow-origin")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(allow_origin, "https://example.test");
    let allow_methods = resp
        .headers()
        .get("access-control-allow-methods")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        allow_methods.to_uppercase().contains("POST"),
        "POST must be in allow-methods: {allow_methods}"
    );
}

/// Without `--cors-allow`, responses do NOT carry an
/// `Access-Control-Allow-Origin` header. The default behavior
/// stays opt-in -- existing clients on the same origin (the
/// `mlpl-serve --static-dir` deploy) are unaffected.
#[tokio::test(flavor = "multi_thread")]
async fn no_cors_flag_means_no_allow_origin_header() {
    let addr = start_server_with_cors(None).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/health"))
        .header("Origin", "https://example.test")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "no CORS flag should not emit Access-Control-Allow-Origin"
    );
}
