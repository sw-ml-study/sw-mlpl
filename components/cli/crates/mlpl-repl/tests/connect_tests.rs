//! Saga 21 step 002: integration tests for the
//! `mlpl-repl --connect <url>` client. Spin
//! `mlpl-serve` up in-process via
//! `mlpl_serve::server::build_app(...)` on a random
//! localhost port, then drive the client through
//! its public HTTP helpers (`create_session`,
//! `eval_remote`, `inspect_remote`). The full
//! interactive `read_loop` is hard to drive
//! programmatically (stdin), so the tests focus on
//! the HTTP layer.
//!
//! The `connect` module lives inside the
//! `mlpl-repl` binary, so the integration test
//! includes its source via `#[path = "..."]`.

// `connect.rs` lives inside the `mlpl-repl` binary;
// re-include it here so the integration test can call
// the public helpers directly. The binary uses every
// item; the test only exercises a subset, so suppress
// dead_code in this compilation unit.
#[allow(dead_code)]
#[path = "../src/connect.rs"]
mod connect;

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::build_app;

async fn start_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(AuthMode::Required);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

fn blocking_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_create_and_eval() {
    let addr = start_server().await;
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = connect::create_session(&client, &base).unwrap();
        assert!(!id.is_empty());
        assert!(!token.is_empty());

        // Eval a couple of programs against the remote
        // session; assert the values come back.
        let r1 = connect::eval_remote(&client, &base, &id, &token, "iota(5)").unwrap();
        assert_eq!(r1.kind, "array");
        assert!(
            r1.value.contains('4'),
            "iota(5) should contain digit 4: {}",
            r1.value
        );

        // State persists.
        connect::eval_remote(&client, &base, &id, &token, "x = 9").unwrap();
        let r3 = connect::eval_remote(&client, &base, &id, &token, "x").unwrap();
        assert!(
            r3.value.contains('9'),
            "x should be 9 after assignment: {}",
            r3.value
        );
    })
    .await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_eval_error_returns_server_error() {
    let addr = start_server().await;
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = connect::create_session(&client, &base).unwrap();

        let err = connect::eval_remote(&client, &base, &id, &token, "undefined_var")
            .expect_err("expected eval error");
        match err {
            connect::ClientError::Server { status, message } => {
                assert_eq!(status, 400, "eval error should be 400");
                assert!(
                    message.to_lowercase().contains("undefined")
                        || message.contains("undefined_var"),
                    "error message should reference the missing variable: {message}"
                );
            }
            other => panic!("expected Server error, got {other:?}"),
        }
    })
    .await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_inspect_returns_workspace_snapshot() {
    let addr = start_server().await;
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = connect::create_session(&client, &base).unwrap();

        // Bind some state on the remote.
        connect::eval_remote(&client, &base, &id, &token, "a = iota(3)").unwrap();
        connect::eval_remote(&client, &base, &id, &token, "b = reshape(iota(6), [2, 3])").unwrap();
        connect::eval_remote(&client, &base, &id, &token, "m = linear(2, 3, 0)").unwrap();

        let snap = connect::inspect_remote(&client, &base, &id, &token).unwrap();
        let var_names: Vec<&str> = snap.vars.iter().map(|v| v.name.as_str()).collect();
        assert!(
            var_names.contains(&"a"),
            "a should be in vars: {var_names:?}"
        );
        assert!(
            var_names.contains(&"b"),
            "b should be in vars: {var_names:?}"
        );
        assert!(
            snap.models.contains(&"m".to_string()),
            "m should be in models: {:?}",
            snap.models
        );
        assert_eq!(snap.more, 0, "no truncation expected");
    })
    .await;
    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_session_create_fails_when_server_down() {
    // Bind a port, immediately drop the listener, then
    // try to create a session against it. The OS will
    // reuse the port slot but nothing is listening --
    // the connect call should fail with a Network
    // error rather than panic.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let err = connect::create_session(&client, &base).expect_err("expected network error");
        assert!(matches!(err, connect::ClientError::Network(_)));
    })
    .await;
    result.unwrap();
}
