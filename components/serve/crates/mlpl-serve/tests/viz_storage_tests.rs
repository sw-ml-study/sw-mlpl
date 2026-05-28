//! Saga 21.5 step 004: viz storage endpoint integration tests.
//!
//! `POST /v1/viz` accepts a base64 payload + content_type and
//! returns a content-addressed URL (`/v1/viz/<sha256-hex-prefix>`).
//! `GET /v1/viz/<id>` returns the same bytes with the recorded
//! `Content-Type` header. The store is in-memory and shared
//! across sessions (the id is the sha256 of the bytes, so
//! identical payloads collide deterministically).
//!
//! Auth: the bearer must match SOME existing session's token
//! when `auth_mode == Required`. Tests cover the happy path,
//! GET 404 on unknown id, missing-bearer 401, and the eval
//! pipeline integration (an SVG-returning program populates the
//! new `viz_url` field on the eval response and the bytes are
//! retrievable through the GET endpoint).

use std::net::SocketAddr;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::build_app;
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
    let id = body["session_id"].as_str().unwrap().to_string();
    let token = body["token"].as_str().unwrap().to_string();
    (id, token)
}

/// `POST /v1/viz` with an SVG payload returns 200 and a URL of
/// the form `/v1/viz/<hex>`. The `id` field is the same hex prefix.
#[tokio::test(flavor = "multi_thread")]
async fn post_viz_returns_content_addressed_url() {
    let addr = start_server(AuthMode::Required).await;
    let (_id, token) = create_session(addr).await;
    let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"10\" height=\"10\"/></svg>";
    let encoded = BASE64.encode(svg.as_bytes());

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/viz"))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "bytes_base64": encoded,
            "content_type": "image/svg+xml",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
    let body: JsonValue = resp.json().await.unwrap();
    let url = body["url"].as_str().unwrap();
    let id = body["id"].as_str().unwrap();
    assert!(
        url.starts_with("/v1/viz/"),
        "url should be /v1/viz/<hex>: {url}"
    );
    assert_eq!(url, format!("/v1/viz/{id}"));
    assert!(id.len() >= 12 && id.chars().all(|c| c.is_ascii_hexdigit()));
}

/// `POST /v1/viz` is content-addressed: two POSTs of the same
/// payload yield the same id.
#[tokio::test(flavor = "multi_thread")]
async fn post_viz_is_content_addressed() {
    let addr = start_server(AuthMode::Required).await;
    let (_id, token) = create_session(addr).await;
    let svg = "<svg><circle r=\"1\"/></svg>";
    let payload = serde_json::json!({
        "bytes_base64": BASE64.encode(svg.as_bytes()),
        "content_type": "image/svg+xml",
    });
    let client = reqwest::Client::new();
    let body1: JsonValue = client
        .post(format!("http://{addr}/v1/viz"))
        .bearer_auth(&token)
        .json(&payload)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let body2: JsonValue = client
        .post(format!("http://{addr}/v1/viz"))
        .bearer_auth(&token)
        .json(&payload)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body1["id"], body2["id"]);
}

/// `GET /v1/viz/<id>` returns the stored bytes with the right
/// `Content-Type` header (passed through verbatim from the POST).
#[tokio::test(flavor = "multi_thread")]
async fn get_viz_returns_stored_bytes_and_content_type() {
    let addr = start_server(AuthMode::Required).await;
    let (_id, token) = create_session(addr).await;
    let svg = "<svg width=\"100\"><line x1=\"0\" y1=\"0\" x2=\"100\" y2=\"100\"/></svg>";
    let post_body: JsonValue = reqwest::Client::new()
        .post(format!("http://{addr}/v1/viz"))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "bytes_base64": BASE64.encode(svg.as_bytes()),
            "content_type": "image/svg+xml",
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let url = post_body["url"].as_str().unwrap();

    let get = reqwest::Client::new()
        .get(format!("http://{addr}{url}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(get.status(), 200);
    let ct = get
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert_eq!(ct, "image/svg+xml");
    let bytes = get.bytes().await.unwrap();
    assert_eq!(bytes.as_ref(), svg.as_bytes());
}

/// `GET /v1/viz/<unknown-id>` returns 404 with the standard
/// `{"error": ...}` shape.
#[tokio::test(flavor = "multi_thread")]
async fn get_viz_unknown_id_is_404() {
    let addr = start_server(AuthMode::Required).await;
    let (_id, token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/viz/deadbeefdead"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// `POST /v1/viz` without a bearer is 401 when auth is required.
#[tokio::test(flavor = "multi_thread")]
async fn post_viz_requires_bearer_when_auth_required() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/viz"))
        .json(&serde_json::json!({
            "bytes_base64": BASE64.encode(b"<svg/>"),
            "content_type": "image/svg+xml",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

/// `GET /v1/viz/<id>` without a bearer is 401 when auth is required.
#[tokio::test(flavor = "multi_thread")]
async fn get_viz_requires_bearer_when_auth_required() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/viz/abc123"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

/// `POST /v1/sessions/<id>/eval` for a program that returns an
/// SVG string now surfaces a `viz_url` field on the response.
/// The URL resolves to the same SVG bytes via `GET /v1/viz/<id>`.
#[tokio::test(flavor = "multi_thread")]
async fn eval_returning_svg_populates_viz_url() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    // `hist(iota(10), 5)` is the cheapest SVG-returning builtin
    // call: histograms render to SVG via mlpl_viz.
    let program = "hist(iota(10), 5)";
    let resp: JsonValue = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": program}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["kind"].as_str().unwrap(), "string");
    let svg_value = resp["value"].as_str().unwrap().to_string();
    let viz_url = resp["viz_url"].as_str().expect("viz_url should be set");
    assert!(
        viz_url.starts_with("/v1/viz/"),
        "viz_url should be a /v1/viz/ path: {viz_url}"
    );

    let get = reqwest::Client::new()
        .get(format!("http://{addr}{viz_url}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(get.status(), 200);
    let bytes = get.bytes().await.unwrap();
    assert_eq!(bytes.as_ref(), svg_value.as_bytes());
}

/// Saga 21.5 step 005: PNG round-trip. The store accepts
/// arbitrary `content_type`; `GET` returns the bytes verbatim
/// with the original `Content-Type` header.
#[tokio::test(flavor = "multi_thread")]
async fn post_and_get_png_preserves_content_type_and_bytes() {
    let addr = start_server(AuthMode::Required).await;
    let (_id, token) = create_session(addr).await;
    // Real PNG signature + a few bytes of payload.
    let png: Vec<u8> = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4];
    let post: JsonValue = reqwest::Client::new()
        .post(format!("http://{addr}/v1/viz"))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "bytes_base64": BASE64.encode(&png),
            "content_type": "image/png",
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let url = post["url"].as_str().unwrap();

    let get = reqwest::Client::new()
        .get(format!("http://{addr}{url}"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(get.status(), 200);
    let ct = get
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(ct, "image/png");
    let bytes = get.bytes().await.unwrap();
    assert_eq!(bytes.as_ref(), png.as_slice());
}

/// Non-SVG eval responses do NOT populate `viz_url` (the detector
/// only fires when the string actually looks like SVG).
#[tokio::test(flavor = "multi_thread")]
async fn eval_returning_non_svg_does_not_populate_viz_url() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let resp: JsonValue = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "iota(5) + 1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        resp.get("viz_url").is_none_or(|v| v.is_null()),
        "viz_url should be absent or null for non-SVG response: {resp:?}"
    );
}
