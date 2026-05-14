//! Saga 21.5 step 003: cancellation endpoint integration tests.
//!
//! `POST /v1/sessions/<id>/cancel` flips a session-scoped
//! `AtomicBool`. The evaluator's loop heads (`for`, `train`,
//! `repeat`) and pre-builtin checkpoints observe it and raise
//! `EvalError::Cancelled`. On the SSE path that lands as a
//! terminal `event: cancelled` carrying the iteration index plus
//! the partial loss curve; the non-streaming `/eval` path returns
//! a `400` with the Display formatting (no structured partials --
//! streaming is the cancel-aware client surface).
//!
//! Server-side double-cancel is idempotent: a second `POST
//! /cancel` after the first is a no-op (the bool is already set).

use std::net::SocketAddr;
use std::time::Duration;

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

/// `POST /cancel` while a `train { }` stream is in flight: the
/// terminal SSE frame is `event: cancelled` carrying the step
/// index and the partial loss curve accumulated so far. The
/// session is re-bindable for further evals after the stream
/// terminates -- `last_losses` survives the cancel.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_mid_train_returns_partial_loss_curve() {
    let addr = start_server().await;
    let (id, token) = create_session(addr).await;
    let client = reqwest::Client::new();
    let eval_url = format!("http://{addr}/v1/sessions/{id}/eval_stream");
    let cancel_url = format!("http://{addr}/v1/sessions/{id}/cancel");

    // 50k iters with a per-iter `reduce_add(iota(500))` body so
    // the whole train is reliably ~100ms+ wall-clock -- gives the
    // cancel POST time to land mid-flight before the loop
    // completes. The last body expression is the metric
    // assignment so `step_val` (and therefore the captured loss)
    // is `step + 0.5`, matching the streamed metric frames.
    let program = "train 50000 { reduce_add(iota(500)) ; loss_metric = step + 0.5 }";
    let eval_client = client.clone();
    let eval_token = token.clone();
    let eval_task = tokio::spawn(async move {
        let resp = eval_client
            .post(eval_url)
            .bearer_auth(eval_token)
            .json(&serde_json::json!({"program": program}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        read_sse_frames(resp).await
    });

    // Let some iterations fire before sending cancel.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let cancel_resp = client
        .post(&cancel_url)
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(
        cancel_resp.status(),
        200,
        "cancel POST should succeed: {:?}",
        cancel_resp.text().await
    );

    let frames = eval_task.await.unwrap();
    let terminal = frames
        .iter()
        .rev()
        .find(|f| matches!(f.event.as_str(), "done" | "cancelled" | "error"))
        .expect("expected a terminal frame");
    assert_eq!(
        terminal.event, "cancelled",
        "expected `cancelled` terminal, got {terminal:?}; all frames: {frames:?}"
    );
    let step = terminal.data["step"].as_u64().expect("step is u64") as usize;
    let partials = terminal.data["partial_losses"]
        .as_array()
        .expect("partial_losses is an array");
    assert!(
        step > 0 && step < 5000,
        "cancel should trip mid-train (0 < step < 5000), got step={step}"
    );
    assert_eq!(
        partials.len(),
        step,
        "partial_losses length ({}) should match step ({step})",
        partials.len()
    );
    for (i, v) in partials.iter().enumerate() {
        let expected = i as f64 + 0.5;
        let actual = v.as_f64().unwrap();
        assert!(
            (actual - expected).abs() < 1e-9,
            "loss[{i}] = {actual}, expected {expected}"
        );
    }
}

/// `POST /cancel` while a `repeat { }` of cheap-but-many builtins
/// is running: the eval returns promptly via the pre-builtin
/// checkpoint, not after the full loop completes.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_mid_builtin_loop_returns_promptly() {
    let addr = start_server().await;
    let (id, token) = create_session(addr).await;
    let client = reqwest::Client::new();
    let eval_url = format!("http://{addr}/v1/sessions/{id}/eval_stream");
    let cancel_url = format!("http://{addr}/v1/sessions/{id}/cancel");

    // Each iteration runs several builtins; the pre-dispatch
    // check trips on the cancel within one op of the bool flip.
    let program = "repeat 100000 { x = reduce_add(iota(50)) ; y = reduce_mul(iota(20)) ; x + y }";
    let eval_client = client.clone();
    let eval_token = token.clone();
    let start = std::time::Instant::now();
    let eval_task = tokio::spawn(async move {
        let resp = eval_client
            .post(eval_url)
            .bearer_auth(eval_token)
            .json(&serde_json::json!({"program": program}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        read_sse_frames(resp).await
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    let cancel_resp = client
        .post(&cancel_url)
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel_resp.status(), 200);

    let frames = eval_task.await.unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "expected prompt return, took {elapsed:?}"
    );
    let terminal = frames
        .iter()
        .rev()
        .find(|f| matches!(f.event.as_str(), "done" | "cancelled" | "error"))
        .expect("terminal frame");
    assert_eq!(
        terminal.event, "cancelled",
        "expected cancelled terminal, got {terminal:?}"
    );
}

/// Double `POST /cancel` is idempotent: the second call sees the
/// bool already set, returns 200 just like the first.
#[tokio::test(flavor = "multi_thread")]
async fn double_cancel_is_idempotent() {
    let addr = start_server().await;
    let (id, token) = create_session(addr).await;
    let cancel_url = format!("http://{addr}/v1/sessions/{id}/cancel");
    let client = reqwest::Client::new();

    let r1 = client
        .post(&cancel_url)
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);
    let r2 = client
        .post(&cancel_url)
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
}

/// `POST /cancel` with a missing bearer token is rejected with
/// 401 (consistent with `/eval`, `/eval_stream`, `/inspect`).
#[tokio::test(flavor = "multi_thread")]
async fn cancel_requires_bearer_token() {
    let addr = start_server().await;
    let (id, _token) = create_session(addr).await;
    let cancel_url = format!("http://{addr}/v1/sessions/{id}/cancel");
    let resp = reqwest::Client::new()
        .post(&cancel_url)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

/// `POST /cancel` for an unknown session id is 404.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_unknown_session_is_404() {
    let addr = start_server().await;
    let bogus = uuid::Uuid::new_v4();
    let cancel_url = format!("http://{addr}/v1/sessions/{bogus}/cancel");
    let resp = reqwest::Client::new()
        .post(&cancel_url)
        .bearer_auth("anything")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}
