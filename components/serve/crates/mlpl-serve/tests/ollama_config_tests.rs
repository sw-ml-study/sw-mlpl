//! Phase 0 (local-gpu-agentic): server-owned Ollama settings.
//! Covers the pure config resolution (flag > env > default + the
//! allow-list) and the two HTTP endpoints: `GET /v1/ollama/config`
//! (report defaults) and `GET /v1/ollama/tags` (allow-listed proxy
//! of the Ollama model list).

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::config::{OllamaConfig, ServeConfig, resolve_ollama};
use mlpl_serve::peers::empty_registry;
use mlpl_serve::server::build_app_with_peers_cors;
use serde_json::{Value, json};

// ---- pure config resolution -------------------------------------

#[test]
fn resolve_precedence_flag_over_env_over_default() {
    let d = OllamaConfig::default();
    // Flag wins over env + default.
    let c = resolve_ollama(
        Some("http://flag:1".into()),
        Some("http://env:2".into()),
        Some("m1".into()),
        vec![],
    );
    assert_eq!(c.default_host, "http://flag:1");
    assert_eq!(c.default_model, "m1");
    // Env wins when no flag; model falls back to default.
    let c = resolve_ollama(None, Some("http://env:2".into()), None, vec![]);
    assert_eq!(c.default_host, "http://env:2");
    assert_eq!(c.default_model, d.default_model);
    // Built-in default when neither flag nor env.
    let c = resolve_ollama(None, None, None, vec![]);
    assert_eq!(c.default_host, d.default_host);
}

#[test]
fn empty_strings_are_ignored_in_resolution() {
    let d = OllamaConfig::default();
    let c = resolve_ollama(
        Some(String::new()),
        Some(String::new()),
        Some(String::new()),
        vec![],
    );
    assert_eq!(c.default_host, d.default_host);
    assert_eq!(c.default_model, d.default_model);
}

#[test]
fn allowed_hosts_cover_default_plus_extras_only() {
    let c = resolve_ollama(
        Some("http://h1:11434".into()),
        None,
        None,
        vec!["http://h2:11434".into(), String::new()],
    );
    assert!(c.is_allowed("http://h1:11434"), "resolved host is allowed");
    assert!(c.is_allowed("http://h2:11434"), "extra host is allowed");
    // Trailing slash is ignored on both sides.
    assert!(c.is_allowed("http://h1:11434/"));
    assert!(!c.is_allowed("http://evil:11434"), "unlisted host rejected");
}

// ---- HTTP endpoints ---------------------------------------------

async fn spawn(ollama: OllamaConfig) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let serve = ServeConfig {
        ollama,
        ..Default::default()
    };
    let app = build_app_with_peers_cors(AuthMode::Disabled, empty_registry(), serve);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

/// A throwaway Ollama stand-in that answers `GET /api/tags`.
async fn spawn_mock_ollama(tags: Value) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/api/tags",
        axum::routing::get(move || {
            let body = tags.clone();
            async move { axum::Json(body) }
        }),
    );
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn config_endpoint_reports_configured_defaults() {
    let ollama = resolve_ollama(
        Some("http://example:11434".into()),
        None,
        Some("smollm2:135m".into()),
        vec![],
    );
    let addr = spawn(ollama).await;
    let body: Value = reqwest::get(format!("http://{addr}/v1/ollama/config"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["host"], "http://example:11434");
    assert_eq!(body["model"], "smollm2:135m");
}

#[tokio::test]
async fn tags_rejects_disallowed_host() {
    let addr = spawn(OllamaConfig::default()).await;
    let resp = reqwest::get(format!(
        "http://{addr}/v1/ollama/tags?host=http://evil:11434"
    ))
    .await
    .unwrap();
    assert_eq!(resp.status(), 403, "disallowed host must be forbidden");
}

#[tokio::test]
async fn tags_proxies_allowlisted_host() {
    let mock = spawn_mock_ollama(json!({"models": [{"name": "smollm2:135m"}]})).await;
    // The mock host is the configured default, so it is allow-listed.
    let ollama = resolve_ollama(Some(mock.clone()), None, None, vec![]);
    let addr = spawn(ollama).await;
    let body: Value = reqwest::get(format!("http://{addr}/v1/ollama/tags"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["models"][0]["name"], "smollm2:135m");
}
