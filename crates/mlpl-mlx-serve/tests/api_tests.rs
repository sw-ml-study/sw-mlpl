//! Saga R1 step 001: REST surface integration tests
//! for `mlpl-mlx-serve`. Spin the server up on a
//! random localhost port via `server::build_app(...)`
//! and a manual `axum::serve(listener, app)`, then
//! drive it with `reqwest`.

use std::net::SocketAddr;

use mlpl_mlx_serve::server::{ServerError, build_app, run};
use mlpl_serve::auth::AuthMode;
use serde_json::Value as JsonValue;

async fn start_server(auth_mode: AuthMode) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(auth_mode);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

#[tokio::test]
async fn health_returns_ok_with_device_mlx() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["device"], "mlx", "device should be mlx");
    assert!(
        !body["version"].as_str().unwrap().is_empty(),
        "version should be set by CARGO_PKG_VERSION"
    );
}

#[tokio::test]
async fn post_sessions_returns_id_and_token() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    let id = body["session_id"].as_str().unwrap();
    let token = body["token"].as_str().unwrap();
    assert!(!id.is_empty(), "session_id must be non-empty");
    assert_eq!(token.len(), 32, "token should be 32 chars");
}

#[tokio::test]
async fn eval_on_device_returns_501_in_step_001() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/eval-on-device"))
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 501);
    let body: JsonValue = resp.json().await.unwrap();
    let err = body["error"].as_str().unwrap();
    assert!(
        err.contains("step 002"),
        "501 body should forward-point at step 002: {err}"
    );
}

#[tokio::test]
async fn run_rejects_non_loopback_with_auth_disabled() {
    // Mirrors the matching test in mlpl-serve; the
    // safety check has the same shape on both
    // services.
    let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    let err = run(addr, AuthMode::Disabled).await.unwrap_err();
    let msg = format!("{err}");
    assert!(
        matches!(err, ServerError::InsecureBind { .. }),
        "expected InsecureBind, got {err:?}"
    );
    assert!(
        msg.contains("--auth required"),
        "error message should mention --auth required: {msg}"
    );
    assert!(
        msg.contains("mlpl-mlx-serve"),
        "error message should identify the binary: {msg}"
    );
}
