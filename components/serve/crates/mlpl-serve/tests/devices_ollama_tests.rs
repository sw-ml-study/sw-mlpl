//! `/v1/devices` reports whether the server's CONFIGURED Ollama host
//! is actually alive, so the web UI can gate the "Ask Ollama" demo on
//! a real LLM backend rather than on mere `?connect=` URL presence.

use std::io::{Read, Write};
use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::config::{OllamaConfig, ServeConfig};
use mlpl_serve::server::build_app_with_peers_cors;
use serde_json::Value as JsonValue;

/// Spin a server whose OllamaConfig points at `host`.
async fn start_with_ollama_host(host: String) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let serve = ServeConfig {
        ollama: OllamaConfig {
            default_host: host.clone(),
            default_model: "test-model".into(),
            allowed_hosts: vec![host],
        },
        ..ServeConfig::default()
    };
    let app = build_app_with_peers_cors(
        AuthMode::Required,
        mlpl_serve::peers::empty_registry(),
        serve,
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// Minimal one-thread HTTP stub that answers every connection with a
/// 200 + `/api/tags`-shaped JSON body, standing in for a live Ollama.
fn spawn_ollama_stub() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            let body = r#"{"models":[]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = s.write_all(resp.as_bytes());
        }
    });
    format!("http://{addr}")
}

async fn fetch_devices_body(addr: SocketAddr) -> JsonValue {
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/devices"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    resp.json().await.unwrap()
}

#[tokio::test]
async fn devices_reports_ollama_false_when_host_is_dead() {
    // Port 9 (discard) on loopback: nothing listens there.
    let addr = start_with_ollama_host("http://127.0.0.1:9".into()).await;
    let body = fetch_devices_body(addr).await;
    assert_eq!(
        body["ollama"], false,
        "dead ollama host must report ollama:false, got {body}"
    );
}

#[tokio::test]
async fn devices_reports_ollama_true_when_host_answers() {
    let stub = spawn_ollama_stub();
    let addr = start_with_ollama_host(stub).await;
    let body = fetch_devices_body(addr).await;
    assert_eq!(
        body["ollama"], true,
        "live ollama host must report ollama:true, got {body}"
    );
}
