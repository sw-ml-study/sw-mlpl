//! Saga 21.5 step 008: `RemoteEvaluator::fetch_viz` tests.
//!
//! When an `/eval` response includes a `viz_url` (because the
//! eval pipeline detected an SVG-returning program), the client
//! GETs `/v1/viz/<id>` to retrieve the raw bytes for rendering
//! inside the browser REPL. This file tests the native impl by
//! spinning `mlpl-serve` up in-process, evaluating a viz
//! producer (`hist`), extracting the URL from the eval response,
//! and asserting the fetched bytes + content-type match the
//! stored value.

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval.rs"]
mod eval;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_native.rs"]
mod eval_native;
#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/frame_trace.rs"]
mod frame_trace;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_sse.rs"]
mod eval_sse;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_native_stream.rs"]
mod eval_native_stream;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_url.rs"]
mod eval_url;

use std::net::SocketAddr;

use eval::RemoteEvaluator;
use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::build_app;
use serde_json::Value as JsonValue;

async fn start_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(AuthMode::Required);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

/// Helper: create a session + run an eval, return the raw JSON
/// response so the test can extract `viz_url` / `value`.
fn eval_program(base: &str, program: &str) -> (String, String, JsonValue) {
    let client = reqwest::blocking::Client::new();
    let create: JsonValue = client
        .post(format!("{base}/v1/sessions"))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let session_id = create["session_id"].as_str().unwrap().to_string();
    let token = create["token"].as_str().unwrap().to_string();
    let body: JsonValue = client
        .post(format!("{base}/v1/sessions/{session_id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": program}))
        .send()
        .unwrap()
        .json()
        .unwrap();
    (session_id, token, body)
}

/// `hist(iota(10), 5)` runs through the eval pipeline, which
/// detects the SVG return value and surfaces `viz_url` on the
/// response. `RemoteEvaluator::fetch_viz` retrieves the same
/// bytes via `GET /v1/viz/<id>`.
#[tokio::test(flavor = "multi_thread")]
async fn fetch_viz_returns_stored_svg_bytes() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let (_id, token, body) = eval_program(&base, "hist(iota(10), 5)");
        let viz_url = body["viz_url"]
            .as_str()
            .expect("hist() should produce viz_url")
            .to_string();
        let svg_value = body["value"].as_str().unwrap().to_string();

        // RemoteEvaluator is constructed with a base URL and a
        // bearer; share the existing session's token because
        // `/v1/viz` auth accepts any known session bearer.
        let remote = RemoteEvaluator::new(&base);
        let (bytes, content_type) = remote
            .fetch_viz(&viz_url, &token)
            .expect("fetch_viz should succeed");
        assert_eq!(content_type, "image/svg+xml");
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            svg_value,
            "fetched bytes should match the eval value"
        );
    })
    .await;
    result.unwrap();
}

/// `fetch_viz` on an unknown URL returns `Err`. The message
/// surfaces the 404 status from the server so callers can map
/// to a useful UI string.
#[tokio::test(flavor = "multi_thread")]
async fn fetch_viz_unknown_url_is_error() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        // Need a valid bearer (any session token works).
        let (_id, token, _body) = eval_program(&base, "iota(3)");
        let remote = RemoteEvaluator::new(&base);
        let err = remote
            .fetch_viz("/v1/viz/deadbeefdeadbeef", &token)
            .expect_err("unknown viz id should fail");
        assert!(
            err.contains("404") || err.to_lowercase().contains("unknown"),
            "error should signal 404 / unknown: {err}"
        );
    })
    .await;
    result.unwrap();
}

/// `fetch_viz` accepts both bare paths (`/v1/viz/<id>`) and
/// fully-qualified URLs (`http://host/v1/viz/<id>`). The eval
/// response surfaces the path form; tests cover the absolute
/// form too since a future browser-bundle deploy might rewrite
/// the URL through a CDN.
#[tokio::test(flavor = "multi_thread")]
async fn fetch_viz_accepts_path_or_absolute_url() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let (_id, token, body) = eval_program(&base, "hist(iota(10), 5)");
        let viz_path = body["viz_url"].as_str().unwrap().to_string();
        let viz_absolute = format!("{base}{viz_path}");
        let remote = RemoteEvaluator::new(&base);
        let (bytes_path, _) = remote
            .fetch_viz(&viz_path, &token)
            .expect("path form should fetch");
        let (bytes_abs, _) = remote
            .fetch_viz(&viz_absolute, &token)
            .expect("absolute form should fetch");
        assert_eq!(bytes_path, bytes_abs);
    })
    .await;
    result.unwrap();
}
