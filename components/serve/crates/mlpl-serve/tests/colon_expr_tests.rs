//! `:name(...)` colon-call expressions must evaluate as programs
//! server-side (user report: `:disp(g)` said "unknown command" /
//! "undefined variable" while `disp(g)` worked), and `:name arg`
//! command-shaped lines over a BUILTIN name get the trichotomy hint.

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::build_app;
use serde_json::{Value as JsonValue, json};

async fn start() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(AuthMode::Required);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

async fn session(addr: SocketAddr) -> (String, String) {
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

async fn eval(addr: SocketAddr, id: &str, tok: &str, program: &str) -> JsonValue {
    reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(tok)
        .json(&json!({ "program": program }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

#[tokio::test]
async fn colon_call_expressions_evaluate_as_programs() {
    let addr = start().await;
    let (id, tok) = session(addr).await;
    eval(addr, &id, &tok, "g = [[0,1],[1,0]]").await;
    let out = eval(addr, &id, &tok, ":disp(g)").await;
    assert_eq!(out["kind"], "string", "expected disp output, got {out}");
    assert!(
        out["value"].as_str().unwrap().contains("rank 2"),
        "disp box expected: {out}"
    );
}

#[tokio::test]
async fn colon_builtin_with_space_gets_the_trichotomy_hint() {
    let addr = start().await;
    let (id, tok) = session(addr).await;
    let out = eval(addr, &id, &tok, ":disp g").await;
    let err = out["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("builtin REFERENCE") && err.contains(":disp(...)"),
        "hint expected, got {err:?}"
    );
}

#[tokio::test]
async fn real_commands_still_reach_inspect() {
    let addr = start().await;
    let (id, tok) = session(addr).await;
    let out = eval(addr, &id, &tok, ":wsid").await;
    assert_eq!(out["kind"], "string", "wsid should answer: {out}");
}
