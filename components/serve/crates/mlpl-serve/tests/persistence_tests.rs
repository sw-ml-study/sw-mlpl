//! Saga 21.5 step 010: session persistence across server
//! restart.
//!
//! When `--persist <path>` is set, the server writes every
//! session's slim state (`session_id`, `token`, timestamps,
//! variables) to a JSON file after each `/eval`. On startup,
//! it reloads the file so a fresh `mlpl-serve` process can
//! pick up where the previous one left off.
//!
//! The slim variant carried in step 010 covers plain numeric
//! variables (the `last_losses` use case + reattach round-trip
//! demos); full Environment serialization -- including models,
//! tokenizers, experiments -- is a follow-up.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::config::ServeConfig;
use mlpl_serve::peers::empty_registry;
use mlpl_serve::server::build_app_with_peers_cors;
use serde_json::Value as JsonValue;
use tempfile::TempDir;

async fn start_server(persist_path: Option<PathBuf>) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app_with_peers_cors(
        AuthMode::Required,
        empty_registry(),
        ServeConfig {
            persist_path,
            ..Default::default()
        },
    );
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
    let id = body["session_id"].as_str().unwrap().to_string();
    let token = body["token"].as_str().unwrap().to_string();
    (id, token)
}

async fn eval_program(addr: SocketAddr, id: &str, token: &str, program: &str) -> JsonValue {
    reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(token)
        .json(&serde_json::json!({"program": program}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// Bind a variable on server A with `--persist <file>`, then
/// spin up server B pointing at the same file. Server B's
/// `GET /v1/sessions/<id>` (with the original token) returns the
/// same metadata and exposes the bound variable.
#[tokio::test(flavor = "multi_thread")]
async fn vars_survive_server_restart_via_persist_path() {
    let tmp = Arc::new(TempDir::new().unwrap());
    let persist_path = tmp.path().join("sessions.json");

    // Server A: bind x = iota(3) + 1.
    let addr_a = start_server(Some(persist_path.clone())).await;
    let (id, token) = create_session(addr_a).await;
    let body = eval_program(addr_a, &id, &token, "x = iota(3) + 1").await;
    assert!(
        body["value"].as_str().is_some(),
        "eval should succeed: {body}"
    );

    // The persist file should exist after the eval (the handler
    // flushes synchronously after a successful eval).
    assert!(
        persist_path.exists(),
        "persist file should be written by eval"
    );

    // Server B: same file, fresh process.
    let addr_b = start_server(Some(persist_path.clone())).await;
    let meta: JsonValue = reqwest::Client::new()
        .get(format!("http://{addr_b}/v1/sessions/{id}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(meta["session_id"].as_str().unwrap(), id);
    let var_names: Vec<&str> = meta["vars"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v["name"].as_str())
        .collect();
    assert!(
        var_names.contains(&"x"),
        "x should survive restart: {var_names:?}"
    );

    // Evaluating `x` on server B returns the same value.
    let eval = eval_program(addr_b, &id, &token, "x").await;
    let v = eval["value"].as_str().unwrap();
    assert!(
        v.contains('1') && v.contains('2') && v.contains('3'),
        "x should round-trip: {v}"
    );
}

/// Without `--persist`, the legacy in-memory-only behavior is
/// preserved: a fresh server doesn't see the prior session.
#[tokio::test(flavor = "multi_thread")]
async fn no_persist_means_no_restart_recovery() {
    let addr_a = start_server(None).await;
    let (id, token) = create_session(addr_a).await;
    eval_program(addr_a, &id, &token, "x = iota(3)").await;

    let addr_b = start_server(None).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr_b}/v1/sessions/{id}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "no persistence -> unknown session");
}

/// Session token round-trips across restart so the original
/// bearer continues to authenticate `GET /v1/sessions/<id>` /
/// `/eval` / etc.
#[tokio::test(flavor = "multi_thread")]
async fn bearer_token_survives_restart() {
    let tmp = TempDir::new().unwrap();
    let persist_path = tmp.path().join("sessions.json");
    let addr_a = start_server(Some(persist_path.clone())).await;
    let (id, token) = create_session(addr_a).await;
    eval_program(addr_a, &id, &token, "x = 7").await;

    let addr_b = start_server(Some(persist_path.clone())).await;
    // Right token: 200.
    let ok = reqwest::Client::new()
        .get(format!("http://{addr_b}/v1/sessions/{id}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), 200);
    // Wrong token: 401.
    let wrong = reqwest::Client::new()
        .get(format!("http://{addr_b}/v1/sessions/{id}"))
        .bearer_auth("not-the-token")
        .send()
        .await
        .unwrap();
    assert_eq!(wrong.status(), 401);
}

/// Malformed / missing persist file does NOT crash startup --
/// the server starts with an empty session map and accepts
/// fresh requests.
#[tokio::test(flavor = "multi_thread")]
async fn malformed_persist_file_is_ignored() {
    let tmp = TempDir::new().unwrap();
    let persist_path = tmp.path().join("garbage.json");
    std::fs::write(&persist_path, "not valid json at all").unwrap();
    let addr = start_server(Some(persist_path)).await;
    // Should still be able to create a fresh session.
    let (id, _token) = create_session(addr).await;
    assert!(!id.is_empty());
}
