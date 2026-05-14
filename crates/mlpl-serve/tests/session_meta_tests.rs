//! Saga 21.5 step 009: `GET /v1/sessions/:id` session-metadata
//! endpoint. Returns a snapshot a client can render before
//! resuming a session: the session id, Unix timestamps for
//! creation + last eval, and the same workspace summary
//! (`vars`, `models`, `tokenizers`, `experiments`, `more`) the
//! existing `/inspect` endpoint surfaces.
//!
//! The reattach client (`mlpl-repl --connect --session <id>
//! --token <tok>`) calls this once on startup to validate the
//! saved credentials AND to show the user what's in the
//! session before they type the first prompt.

use std::net::SocketAddr;

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

/// Right after `POST /v1/sessions`, `GET /v1/sessions/:id`
/// returns metadata: the same session_id, a `created_at`
/// timestamp, no `last_eval_at` yet (session is fresh), and an
/// empty workspace summary.
#[tokio::test(flavor = "multi_thread")]
async fn get_session_returns_metadata_for_fresh_session() {
    let addr = start_server().await;
    let (id, token) = create_session(addr).await;
    let body: JsonValue = reqwest::Client::new()
        .get(format!("http://{addr}/v1/sessions/{id}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["session_id"].as_str().unwrap(), id);
    assert!(
        body["created_at"].as_u64().is_some(),
        "created_at should be a Unix timestamp: {body}"
    );
    assert!(body["last_eval_at"].is_null(), "no eval ran yet: {body}");
    assert!(
        body["vars"].as_array().unwrap().is_empty(),
        "fresh session: no vars: {body}"
    );
}

/// After an eval that binds variables, the metadata reflects
/// the new vars AND populates `last_eval_at` with a timestamp
/// at-or-after `created_at`.
#[tokio::test(flavor = "multi_thread")]
async fn get_session_reflects_vars_and_last_eval_after_eval() {
    let addr = start_server().await;
    let (id, token) = create_session(addr).await;
    let client = reqwest::Client::new();
    let _eval: JsonValue = client
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "x = iota(3)"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let meta: JsonValue = client
        .get(format!("http://{addr}/v1/sessions/{id}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let created = meta["created_at"].as_u64().unwrap();
    let last_eval = meta["last_eval_at"]
        .as_u64()
        .expect("last_eval_at should be populated after eval");
    assert!(
        last_eval >= created,
        "last_eval_at {last_eval} should be >= created_at {created}"
    );
    let var_names: Vec<&str> = meta["vars"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v["name"].as_str())
        .collect();
    assert!(
        var_names.contains(&"x"),
        "x should be in vars: {var_names:?}"
    );
}

/// Unknown session id is 404.
#[tokio::test(flavor = "multi_thread")]
async fn get_session_unknown_id_is_404() {
    let addr = start_server().await;
    let bogus = uuid::Uuid::new_v4();
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/sessions/{bogus}"))
        .bearer_auth("anything")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// Missing bearer is 401 when auth is Required.
#[tokio::test(flavor = "multi_thread")]
async fn get_session_requires_bearer_when_auth_required() {
    let addr = start_server().await;
    let (id, _token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/sessions/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

/// Wrong bearer (matches no session) is 401.
#[tokio::test(flavor = "multi_thread")]
async fn get_session_wrong_bearer_is_401() {
    let addr = start_server().await;
    let (id, _token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/sessions/{id}"))
        .bearer_auth("not-the-real-token")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}
