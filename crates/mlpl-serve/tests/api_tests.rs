//! Saga 21 step 001: REST surface integration
//! tests. Spin `mlpl-serve` up on a random localhost
//! port via `server::build_app(...)` + a manual
//! `axum::serve(listener, app)`, then drive it with
//! `reqwest`.

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::{ServerError, build_app, run};
use serde_json::Value as JsonValue;

/// Spin up a server in the background on a random
/// loopback port. Returns the bound address; the
/// task runs for the duration of the test.
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
    assert_eq!(resp.status(), 200, "session create should be 200");
    let body: JsonValue = resp.json().await.unwrap();
    let id = body["session_id"].as_str().unwrap().to_string();
    let token = body["token"].as_str().unwrap().to_string();
    assert!(!id.is_empty(), "session_id must be non-empty");
    assert!(!token.is_empty(), "token must be non-empty");
    (id, token)
}

#[tokio::test]
async fn post_sessions_returns_id_and_token() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    // 32 alphanumeric is the contract; check both the
    // length and that no two sessions reuse a token.
    assert_eq!(token.len(), 32, "token should be 32 chars");
    let (id2, token2) = create_session(addr).await;
    assert_ne!(id, id2, "session ids must be unique");
    assert_ne!(token, token2, "tokens must be unique");
}

#[tokio::test]
async fn eval_runs_program_against_session_env() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "iota(5)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["kind"], "array", "iota returns an array");
    let value = body["value"].as_str().unwrap();
    // iota(5) prints as something containing 0..4 in
    // some bracketed form; we don't pin the exact
    // formatting, but the digits must be there.
    for digit in ['0', '1', '2', '3', '4'] {
        assert!(
            value.contains(digit),
            "iota(5) value {value:?} should contain digit {digit}"
        );
    }
}

#[tokio::test]
async fn eval_persists_state_across_calls() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let url = format!("http://{addr}/v1/sessions/{id}/eval");
    let client = reqwest::Client::new();

    // First call binds `x = 7` (assignment returns
    // a sentinel; we don't care about the value).
    let r1 = client
        .post(&url)
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "x = 7"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);

    // Second call reads `x` -- proves the env
    // survived between requests.
    let r2 = client
        .post(&url)
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let body: JsonValue = r2.json().await.unwrap();
    assert!(
        body["value"].as_str().unwrap().contains('7'),
        "second eval should see x=7 from the first call"
    );
}

#[tokio::test]
async fn eval_without_bearer_returns_401() {
    let addr = start_server(AuthMode::Required).await;
    let (id, _token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
    let body: JsonValue = resp.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("authorization"),
        "401 body should mention authorization"
    );
}

#[tokio::test]
async fn eval_with_wrong_bearer_returns_401() {
    let addr = start_server(AuthMode::Required).await;
    let (id, _token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth("not-the-real-token-xxxxxxxxxxxxx")
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn eval_unknown_session_returns_404() {
    let addr = start_server(AuthMode::Required).await;
    let bogus = "00000000-0000-0000-0000-000000000000";

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{bogus}/eval"))
        .bearer_auth("anything")
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn eval_program_error_returns_400_with_message() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "undefined_var_xyz"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: JsonValue = resp.json().await.unwrap();
    let err = body["error"].as_str().unwrap();
    assert!(
        err.contains("undefined_var_xyz") || err.to_lowercase().contains("undefined"),
        "400 body should reference the missing variable: {err}"
    );
}

#[tokio::test]
async fn health_returns_ok_and_version() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert!(
        !body["version"].as_str().unwrap().is_empty(),
        "version should be set by CARGO_PKG_VERSION"
    );
}

#[tokio::test]
async fn run_rejects_non_loopback_with_auth_disabled() {
    // Pick an arbitrary non-loopback address; we
    // don't actually expect to bind it. The safety
    // check should fail before bind.
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
}

#[tokio::test]
async fn auth_disabled_skips_bearer_check() {
    let addr = start_server(AuthMode::Disabled).await;
    let (id, _token) = create_session(addr).await;

    // Note: NO bearer header.
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "with AuthMode::Disabled, missing bearer should still succeed"
    );
}
