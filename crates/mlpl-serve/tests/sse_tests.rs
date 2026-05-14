//! Saga 21.5 step 001: SSE streaming eval (`/eval_stream`)
//! integration tests. Spin the server up on a random loopback port
//! the same way `api_tests.rs` does, then POST programs and read
//! the SSE response body chunk-by-chunk.

use std::net::SocketAddr;

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

/// One SSE frame: an `event:` name + its `data:` payload (parsed
/// from the body JSON when non-empty). Built by `read_sse_frames`
/// from a streaming response body. Drops `:` comment lines (keep-
/// alive pings) so test assertions only see real events.
#[derive(Debug)]
struct SseFrame {
    event: String,
    data: JsonValue,
}

async fn read_sse_frames(resp: reqwest::Response) -> Vec<SseFrame> {
    use tokio_stream::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.unwrap();
        buf.extend_from_slice(&bytes);
    }
    let text = String::from_utf8(buf).unwrap();
    let mut frames = Vec::new();
    for block in text.split("\n\n") {
        if block.trim().is_empty() {
            continue;
        }
        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim().to_string());
            }
            // `:` comment lines (axum's keep-alive) are skipped.
        }
        let event = match event_name {
            Some(e) => e,
            None => continue,
        };
        let data_str = data_lines.join("\n");
        let data = if data_str.is_empty() {
            JsonValue::Null
        } else {
            serde_json::from_str(&data_str).unwrap_or(JsonValue::Null)
        };
        frames.push(SseFrame { event, data });
    }
    frames
}

#[tokio::test]
async fn eval_stream_runs_simple_program_and_returns_done() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval_stream"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "iota(5) + 1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );

    let frames = read_sse_frames(resp).await;
    assert!(!frames.is_empty(), "expected at least one SSE frame");
    assert_eq!(frames[0].event, "ready", "first frame must be `ready`");
    // No `_metric` bindings in this program -> no metric frames.
    let metrics: Vec<&SseFrame> = frames.iter().filter(|f| f.event == "metric").collect();
    assert!(metrics.is_empty(), "expected zero metric frames");
    // Terminal frame is `done`, never both done + error.
    let terminals: Vec<&SseFrame> = frames
        .iter()
        .filter(|f| f.event == "done" || f.event == "error")
        .collect();
    assert_eq!(terminals.len(), 1, "expected exactly one terminal frame");
    let done = terminals[0];
    assert_eq!(done.event, "done");
    assert_eq!(done.data["kind"], "array");
    let value = done.data["value"].as_str().unwrap();
    for digit in ['1', '2', '3', '4', '5'] {
        assert!(
            value.contains(digit),
            "done.value {value:?} should contain {digit}"
        );
    }
}

#[tokio::test]
async fn eval_stream_done_payload_matches_eval_response() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let client = reqwest::Client::new();
    let program = "iota(7)";

    let plain = client
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({ "program": program }))
        .send()
        .await
        .unwrap();
    let plain_body: JsonValue = plain.json().await.unwrap();

    let stream_resp = client
        .post(format!("http://{addr}/v1/sessions/{id}/eval_stream"))
        .bearer_auth(&token)
        .json(&serde_json::json!({ "program": program }))
        .send()
        .await
        .unwrap();
    let frames = read_sse_frames(stream_resp).await;
    let done = frames.iter().find(|f| f.event == "done").unwrap();

    // `done.data` payload must match `/eval`'s body byte-for-byte
    // on the shared fields. Schema-wise the SSE form carries the
    // same `value` + `kind` keys as `EvalResponse`.
    assert_eq!(done.data["value"], plain_body["value"]);
    assert_eq!(done.data["kind"], plain_body["kind"]);
}

#[tokio::test]
async fn eval_stream_emits_metric_events_during_train() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    // 5-iteration train block where the body assigns
    // `loss_metric = step + 0.5` so the per-iteration scalar is
    // predictable. Each iteration ends with a non-metric
    // expression (the assignment's value is the final body
    // expression so the iteration completes cleanly).
    let program = "train 5 { loss_metric = step + 0.5 ; step + 0.5 }";
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval_stream"))
        .bearer_auth(&token)
        .json(&serde_json::json!({ "program": program }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let frames = read_sse_frames(resp).await;

    assert_eq!(frames[0].event, "ready");
    let metrics: Vec<&SseFrame> = frames.iter().filter(|f| f.event == "metric").collect();
    assert_eq!(
        metrics.len(),
        5,
        "expected 5 metric frames, got {metrics:?}"
    );
    for (i, m) in metrics.iter().enumerate() {
        assert_eq!(m.data["name"], "loss_metric");
        assert_eq!(m.data["step"].as_u64().unwrap(), i as u64);
        let v = m.data["value"].as_f64().unwrap();
        assert!(
            (v - (i as f64 + 0.5)).abs() < 1e-9,
            "iter {i} value {v} should be {}",
            i as f64 + 0.5
        );
    }
    assert_eq!(frames.last().unwrap().event, "done");
}

#[tokio::test]
async fn eval_stream_without_bearer_returns_401() {
    let addr = start_server(AuthMode::Required).await;
    let (id, _token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval_stream"))
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn eval_stream_unknown_session_returns_404() {
    let addr = start_server(AuthMode::Required).await;
    let bogus = "00000000-0000-0000-0000-000000000000";

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{bogus}/eval_stream"))
        .bearer_auth("anything")
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn eval_stream_runtime_error_emits_error_frame() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    // Reference an undefined identifier -- lex + parse succeed,
    // so we get past the up-front 400 path and into the eval
    // task. The eval task should emit `event: error`.
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval_stream"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "undefined_var_xyz"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "lex/parse succeeded; SSE opened");
    let frames = read_sse_frames(resp).await;
    assert_eq!(frames[0].event, "ready");
    let terminal = frames.last().unwrap();
    assert_eq!(terminal.event, "error");
    let err = terminal.data["error"].as_str().unwrap();
    assert!(
        err.to_lowercase().contains("undefined") || err.contains("undefined_var_xyz"),
        "error payload should reference the missing variable, got {err:?}"
    );
}

#[tokio::test]
async fn eval_stream_lex_error_returns_400_plain_json() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval_stream"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "@@@"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "lex/parse errors surface as plain JSON 400, not SSE error frames"
    );
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.starts_with("application/json"),
        "expected JSON body for lex/parse error, got {ct:?}"
    );
}
