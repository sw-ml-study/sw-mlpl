//! Saga 21.5 step 009: `mlpl-repl --connect --session <id>
//! --token <tok>` reattach.
//!
//! Tests the pure pieces:
//!   1. `parse_connect_args` recognizes `--session` + `--token`
//!      flags and produces a `Reattach { session_id, token }`
//!      payload on `ConnectMode::Remote`.
//!   2. `read_persisted_session` / `write_persisted_session`
//!      round-trip a session id + token through a JSON file
//!      pointed at by `MLPL_REPL_SESSION_FILE`.
//!   3. `fetch_session_meta` against an in-process `mlpl-serve`
//!      returns the metadata snapshot a reattach session uses.

#[allow(dead_code)]
#[path = "../src/connect.rs"]
mod connect;

#[allow(dead_code)]
#[path = "../src/connect_reattach.rs"]
mod connect_reattach;

#[allow(dead_code)]
#[path = "../src/connect_stream.rs"]
mod connect_stream;

use std::net::SocketAddr;

use connect_stream::{ConnectMode, parse_connect_args};
use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::build_app;
use tempfile::TempDir;

async fn start_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(AuthMode::Required);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

// ---- parse_connect_args ----

#[test]
fn parse_session_and_token_yields_reattach() {
    let args = [
        "mlpl-repl",
        "--connect",
        "http://localhost:6464",
        "--session",
        "deadbeef",
        "--token",
        "abc123",
    ]
    .iter()
    .map(ToString::to_string)
    .collect::<Vec<_>>();
    let mode = parse_connect_args(&args, None).unwrap();
    match mode {
        ConnectMode::Remote {
            url,
            stream,
            reattach,
        } => {
            assert_eq!(url, "http://localhost:6464");
            assert!(!stream);
            let r = reattach.expect("--session + --token should produce reattach");
            assert_eq!(r.session_id, "deadbeef");
            assert_eq!(r.token, "abc123");
        }
        other => panic!("expected Remote, got {other:?}"),
    }
}

#[test]
fn parse_no_reattach_without_session_or_token() {
    let args = [
        "mlpl-repl".to_string(),
        "--connect".to_string(),
        "http://x".to_string(),
    ];
    let mode = parse_connect_args(&args, None).unwrap();
    match mode {
        ConnectMode::Remote { reattach, .. } => assert!(reattach.is_none()),
        other => panic!("expected Remote, got {other:?}"),
    }
}

#[test]
fn parse_session_without_token_errors() {
    let args = [
        "mlpl-repl".to_string(),
        "--connect".to_string(),
        "http://x".to_string(),
        "--session".to_string(),
        "deadbeef".to_string(),
    ];
    let err = parse_connect_args(&args, None).unwrap_err();
    assert!(
        matches!(err, connect_stream::ConnectArgsError::ReattachIncomplete),
        "expected ReattachIncomplete, got {err:?}"
    );
}

#[test]
fn parse_token_without_session_errors() {
    let args = [
        "mlpl-repl".to_string(),
        "--connect".to_string(),
        "http://x".to_string(),
        "--token".to_string(),
        "abc".to_string(),
    ];
    let err = parse_connect_args(&args, None).unwrap_err();
    assert!(matches!(
        err,
        connect_stream::ConnectArgsError::ReattachIncomplete
    ));
}

// ---- persisted-session file ----

#[test]
fn write_then_read_persisted_session_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("session.json");
    connect_reattach::write_persisted_session(&path, "session-123", "token-xyz").unwrap();
    let (id, token) = connect_reattach::read_persisted_session(&path).unwrap();
    assert_eq!(id, "session-123");
    assert_eq!(token, "token-xyz");
}

#[test]
fn read_persisted_session_missing_file_is_none() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("does-not-exist.json");
    assert!(connect_reattach::read_persisted_session(&path).is_none());
}

#[test]
fn read_persisted_session_malformed_is_none() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("malformed.json");
    std::fs::write(&path, "not json at all").unwrap();
    assert!(connect_reattach::read_persisted_session(&path).is_none());
}

// ---- fetch_session_meta against live server ----

/// `connect_reattach::fetch_session_meta` returns Ok for an
/// existing session with the right token and Err for a wrong
/// token. Used by reattach to validate the cached credentials
/// before printing the welcome banner.
#[tokio::test(flavor = "multi_thread")]
async fn fetch_session_meta_validates_credentials() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let client = connect::build_client();
        // Create a session via the existing helper.
        let (id, token) = connect::create_session(&client, &base).unwrap();

        // Right token -> Ok with metadata.
        let meta = connect_reattach::fetch_session_meta(&client, &base, &id, &token)
            .expect("right token should succeed");
        assert_eq!(meta.session_id, id);
        assert!(meta.created_at > 0);

        // Wrong token -> Err.
        let err = connect_reattach::fetch_session_meta(&client, &base, &id, "not-the-token")
            .expect_err("wrong token should fail");
        match err {
            connect::ClientError::Server { status, .. } => assert_eq!(status, 401),
            other => panic!("expected Server 401, got {other:?}"),
        }
    })
    .await;
    result.unwrap();
}
