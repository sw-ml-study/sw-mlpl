//! Saga 21.5 step 003: cancellation in `mlpl-repl --connect`.
//!
//! The interactive REPL binds SIGINT to a debounced "press twice
//! to cancel" handler: the first Ctrl-C is a warning, the second
//! Ctrl-C within a short window POSTs `/v1/sessions/<id>/cancel`
//! and surfaces the partial loss curve back to the user.
//!
//! Tests target the pure pieces:
//!   1. `should_double_cancel` -- the debounce predicate.
//!   2. `post_cancel` -- the cancel-HTTP helper.
//!   3. Streaming a `train` against an in-process `mlpl-serve`,
//!      POSTing `/cancel` mid-flight, then asserting that the
//!      returned `ClientError::Cancelled` carries the partial
//!      losses (mirror of the SSE wire test, exercised through
//!      the client API the REPL actually uses).

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
use std::time::{Duration, Instant};

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
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

/// First Ctrl-C with no prior press -- treat as a warning, do
/// NOT cancel yet.
#[test]
fn first_ctrl_c_is_not_a_cancel() {
    let now = Instant::now();
    let window = Duration::from_secs(2);
    assert!(!connect_stream::should_double_cancel(None, now, window));
}

/// Second Ctrl-C within the window -- cancel.
#[test]
fn second_ctrl_c_within_window_is_a_cancel() {
    let now = Instant::now();
    let window = Duration::from_secs(2);
    let prev = now - Duration::from_millis(500);
    assert!(connect_stream::should_double_cancel(
        Some(prev),
        now,
        window
    ));
}

/// Second Ctrl-C after the window has elapsed -- not a cancel
/// (treat as a fresh first press; the next one within window
/// would cancel).
#[test]
fn second_ctrl_c_outside_window_is_not_a_cancel() {
    let now = Instant::now();
    let window = Duration::from_secs(2);
    let prev = now - Duration::from_secs(5);
    assert!(!connect_stream::should_double_cancel(
        Some(prev),
        now,
        window
    ));
}

/// `post_cancel` against a live server returns Ok and lands a 200.
#[tokio::test(flavor = "multi_thread")]
async fn post_cancel_against_live_server_returns_ok() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = connect::create_session(&client, &base).unwrap();
        connect_stream::post_cancel(&client, &base, &id, &token).unwrap();
        // Idempotent: a second cancel still returns Ok.
        connect_stream::post_cancel(&client, &base, &id, &token).unwrap();
    })
    .await;
    result.unwrap();
}

/// Cancel mid-train via the streaming client API: the partial
/// loss curve comes back as `ClientError::Cancelled` carrying the
/// step index and the per-iteration losses. Mirrors the server-side
/// `sse_tests` cancel test but exercises the client helpers the
/// REPL actually uses, so a regression in the client SSE parser
/// would be caught here too.
#[tokio::test(flavor = "multi_thread")]
async fn streaming_train_cancel_surfaces_partial_losses_to_client() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = connect::create_session(&client, &base).unwrap();

        // Spawn the cancel from a side thread shortly after eval
        // starts. The streaming call blocks until the server
        // emits the terminal `cancelled` frame.
        let base_c = base.clone();
        let id_c = id.clone();
        let token_c = token.clone();
        let client_c = client.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            let _ = connect_stream::post_cancel(&client_c, &base_c, &id_c, &token_c);
        });

        let mut frames: Vec<(usize, f64)> = Vec::new();
        let err = connect_stream::eval_remote_stream(
            &client,
            &base,
            &id,
            &token,
            // Beefier body (`reduce_add(iota(500))` per iter) and
            // 50k iters so the train is reliably ~100ms wall-clock,
            // giving the spawned cancel POST time to land
            // mid-flight before the loop completes. Last body
            // expression is the metric assignment so `step_val`
            // matches the streamed metric value.
            "train 50000 { reduce_add(iota(500)) ; loss_metric = step + 0.5 }",
            &mut |m| frames.push((m.step, m.value)),
        )
        .expect_err("expected cancellation, got success");

        match err {
            connect::ClientError::Cancelled {
                step,
                partial_losses,
            } => {
                assert!(
                    step > 0 && step < 50000,
                    "cancel should land mid-train, step={step}"
                );
                assert_eq!(
                    partial_losses.len(),
                    step,
                    "partial_losses len {} != step {step}",
                    partial_losses.len()
                );
                for (i, v) in partial_losses.iter().enumerate() {
                    let expected = i as f64 + 0.5;
                    assert!(
                        (v - expected).abs() < 1e-9,
                        "loss[{i}] = {v}, expected {expected}"
                    );
                }
            }
            other => panic!("expected ClientError::Cancelled, got {other:?}"),
        }
    })
    .await;
    result.unwrap();
}
