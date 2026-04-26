//! Saga R1 step 001 + 002: REST surface integration
//! tests for `mlpl-mlx-serve`. Spin the server up
//! on a random localhost port via
//! `server::build_app(...)` and a manual
//! `axum::serve(listener, app)`, then drive it with
//! `reqwest`. Step 002 added eval-on-device,
//! transfer, and release-handle endpoints on top of
//! the wire format module.

use std::net::SocketAddr;

use mlpl_array::{DenseArray, Shape};
use mlpl_mlx_serve::server::{ServerError, build_app, run};
use mlpl_mlx_serve::wire::{decode_from_json, encode_for_json};
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

async fn create_session(addr: SocketAddr) -> (String, String) {
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .send()
        .await
        .unwrap();
    let body: JsonValue = resp.json().await.unwrap();
    (
        body["session_id"].as_str().unwrap().to_string(),
        body["token"].as_str().unwrap().to_string(),
    )
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
async fn eval_on_device_no_bindings_returns_tensor_handle() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval-on-device"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "iota(5)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["result"]["kind"], "tensor");
    assert_eq!(body["result"]["device"], "mlx");
    assert_eq!(body["result"]["shape"], serde_json::json!([5]));
    assert!(!body["result"]["handle"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn eval_on_device_with_bindings_uses_them() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let x = DenseArray::new(Shape::new(vec![3]), vec![1.0, 2.0, 3.0]).unwrap();
    let body = serde_json::json!({
        "program": "x * 2",
        "bindings": [{"name": "x", "tensor": encode_for_json(&x).unwrap()}],
    });
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval-on-device"))
        .bearer_auth(&token)
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let resp_body: JsonValue = resp.json().await.unwrap();
    assert_eq!(resp_body["result"]["kind"], "tensor");
    assert_eq!(resp_body["result"]["shape"], serde_json::json!([3]));

    // Transfer the handle back; verify x*2 contents.
    let handle = resp_body["result"]["handle"].as_str().unwrap();
    let tx = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/transfer"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"handle": handle}))
        .send()
        .await
        .unwrap();
    assert_eq!(tx.status(), 200);
    let tx_body: JsonValue = tx.json().await.unwrap();
    let bytes_str = tx_body["tensor"].as_str().unwrap();
    let recovered = decode_from_json(bytes_str).unwrap();
    let expected = DenseArray::new(Shape::new(vec![3]), vec![2.0, 4.0, 6.0]).unwrap();
    assert_eq!(recovered, expected);
}

#[tokio::test]
async fn release_handle_then_transfer_returns_404() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/v1/sessions/{id}/eval-on-device"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    let body: JsonValue = resp.json().await.unwrap();
    let handle = body["result"]["handle"].as_str().unwrap().to_string();

    let release = client
        .post(format!(
            "http://{addr}/v1/sessions/{id}/release-handle/{handle}"
        ))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(release.status(), 204);

    let tx = client
        .post(format!("http://{addr}/v1/sessions/{id}/transfer"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"handle": handle}))
        .send()
        .await
        .unwrap();
    assert_eq!(tx.status(), 404);
}

#[tokio::test]
async fn eval_on_device_no_bearer_returns_401() {
    let addr = start_server(AuthMode::Required).await;
    let (id, _token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval-on-device"))
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn eval_on_device_unknown_session_returns_404() {
    let addr = start_server(AuthMode::Required).await;
    let bogus = "00000000-0000-0000-0000-000000000000";
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{bogus}/eval-on-device"))
        .bearer_auth("anything")
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn eval_on_device_eval_error_returns_400() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval-on-device"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "undefined_var_xyz"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: JsonValue = resp.json().await.unwrap();
    let err = body["error"].as_str().unwrap();
    assert!(
        err.to_lowercase().contains("undefined") || err.contains("undefined_var_xyz"),
        "400 body should reference the missing variable: {err}"
    );
}

#[tokio::test]
async fn eval_on_device_malformed_binding_returns_400() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval-on-device"))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "program": "x",
            "bindings": [{"name": "x", "tensor": "not-base64-or-bincode"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
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
