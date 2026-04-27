//! Saga R1 step 004: orchestrator peer routing. The
//! peer here is a minimal contract-compatible HTTP
//! service so this crate can test dispatch without a
//! Cargo cycle back through `mlpl-mlx-serve`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::Path;
use axum::routing::post;
use axum::{Json, Router};
use mlpl_serve::auth::AuthMode;
use mlpl_serve::peers::build_registry;
use mlpl_serve::server::build_app_with_peers;
use serde_json::Value as JsonValue;

#[derive(Clone, Default)]
struct PeerState {
    calls: Arc<Mutex<Vec<JsonValue>>>,
}

async fn start_orchestrator(peer_addr: SocketAddr) -> SocketAddr {
    let peers = build_registry(vec![("mlx".into(), format!("http://{peer_addr}"))], false).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app_with_peers(AuthMode::Required, peers);
    listener.set_nonblocking(true).unwrap();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    addr
}

async fn start_peer() -> (SocketAddr, PeerState) {
    let state = PeerState::default();
    let app = Router::new()
        .route("/v1/sessions", post(create_peer_session))
        .route("/v1/sessions/:id/eval-on-device", post(eval_on_device))
        .route("/v1/sessions/:id/transfer", post(transfer))
        .with_state(state.clone());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    (addr, state)
}

async fn create_peer_session() -> Json<JsonValue> {
    Json(serde_json::json!({
        "session_id": "00000000-0000-0000-0000-000000000001",
        "token": "peer-token",
    }))
}

async fn eval_on_device(
    axum::extract::State(state): axum::extract::State<PeerState>,
    Path(_id): Path<String>,
    Json(body): Json<JsonValue>,
) -> Json<JsonValue> {
    state.calls.lock().unwrap().push(body);
    Json(serde_json::json!({
        "result": {
            "kind": "tensor",
            "handle": "h1",
            "shape": [3],
            "device": "mlx"
        }
    }))
}

async fn transfer(Path(_id): Path<String>, Json(_body): Json<JsonValue>) -> Json<JsonValue> {
    Json(serde_json::json!({
        "tensor": encode_iota3(),
    }))
}

fn encode_iota3() -> String {
    // Same envelope as the R1 wire contract:
    // version=1, dtype=0(f64), ndim=1, shape=[3].
    let env = WireEnvelope {
        version: 1,
        dtype: 0,
        ndim: 1,
        shape: vec![3],
        data: [0.0_f64, 1.0, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    };
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bincode::serialize(&env).unwrap())
}

#[derive(serde::Serialize)]
struct WireEnvelope {
    version: u32,
    dtype: u8,
    ndim: u8,
    shape: Vec<u64>,
    data: Vec<u8>,
}

async fn create_session(addr: SocketAddr) -> (String, String) {
    let body: JsonValue = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    (
        body["session_id"].as_str().unwrap().to_string(),
        body["token"].as_str().unwrap().to_string(),
    )
}

async fn eval(addr: SocketAddr, id: &str, token: &str, program: &str) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(token)
        .json(&serde_json::json!({ "program": program }))
        .send()
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn device_block_routes_to_peer_and_fetches_back_to_cpu() {
    let (peer_addr, peer_state) = start_peer().await;
    let orch_addr = start_orchestrator(peer_addr).await;
    let (id, token) = create_session(orch_addr).await;

    let remote = eval(orch_addr, &id, &token, "x = device(\"mlx\") { iota(3) }").await;
    assert_eq!(remote.status(), 200);
    let remote_body: JsonValue = remote.json().await.unwrap();
    assert_eq!(remote_body["kind"], "device-tensor");
    assert!(remote_body["value"].as_str().unwrap().contains("shape=[3]"));

    let fault = eval(orch_addr, &id, &token, "x + 1").await;
    assert_eq!(fault.status(), 400);
    let fault_body: JsonValue = fault.json().await.unwrap();
    assert!(
        fault_body["error"]
            .as_str()
            .unwrap()
            .contains("to_device('cpu', x)")
    );

    let fetched = eval(orch_addr, &id, &token, "to_device(\"cpu\", x)").await;
    assert_eq!(fetched.status(), 200);
    let fetched_body: JsonValue = fetched.json().await.unwrap();
    assert_eq!(fetched_body["kind"], "array");
    for digit in ["0", "1", "2"] {
        assert!(fetched_body["value"].as_str().unwrap().contains(digit));
    }

    let calls = peer_state.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["bindings"], serde_json::json!([]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn device_block_without_peer_falls_back_to_existing_behavior() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app_with_peers(AuthMode::Required, Arc::new(HashMap::new()));
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let (id, token) = create_session(addr).await;

    let resp = eval(addr, &id, &token, "device(\"mlx\") { iota(3) }").await;
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["kind"], "array");
}
