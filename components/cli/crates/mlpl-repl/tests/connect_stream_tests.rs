//! Saga 21.5 step 002: integration tests for
//! `mlpl-repl --connect <url> --stream`. The streaming
//! eval path POSTs to `/v1/sessions/<id>/eval_stream`
//! (SSE) and renders one progress line per `_metric`
//! frame; the existing non-streaming path stays as a
//! default for quiet scripts.
//!
//! `connect_stream.rs` lives inside the `mlpl-repl`
//! binary; re-include it (and `connect.rs`, which it
//! depends on for `ClientError`/`EvalResponse`) via
//! `#[path = "..."]` so the integration test can call
//! the public helpers directly.

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

/// Streaming `iota(5) + 1` produces the same final value
/// as the non-streaming `/eval` path, byte-for-byte. The
/// SSE stream emits no `metric` frames for a non-`train`
/// program, so the callback is never invoked.
#[tokio::test(flavor = "multi_thread")]
async fn stream_and_non_stream_produce_same_final_value() {
    let addr = start_server().await;
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = mlpl_repl_connect::connect::create_session(&client, &base).unwrap();

        let r_nonstream =
            mlpl_repl_connect::connect::eval_remote(&client, &base, &id, &token, "iota(5) + 1")
                .unwrap();

        let mut metrics: Vec<mlpl_repl_connect::connect_stream::SseMetric> = Vec::new();
        let r_stream = mlpl_repl_connect::connect_stream::eval_remote_stream(
            &client,
            &base,
            &id,
            &token,
            "iota(5) + 1",
            &mut |m| {
                metrics.push(mlpl_repl_connect::connect_stream::SseMetric {
                    name: m.name.clone(),
                    step: m.step,
                    value: m.value,
                })
            },
        )
        .unwrap();

        assert_eq!(r_stream.value, r_nonstream.value);
        assert_eq!(r_stream.kind, r_nonstream.kind);
        assert!(
            metrics.is_empty(),
            "iota(5) + 1 should emit no _metric frames, got {metrics:?}"
        );
    })
    .await;
    result.unwrap();
}

/// `train N { loss_metric = ... }` produces one
/// `event: metric` frame per iteration. The streaming
/// client surfaces each as a callback with the correct
/// `(name, step, value)` triple.
#[tokio::test(flavor = "multi_thread")]
async fn stream_emits_one_metric_per_train_iteration() {
    let addr = start_server().await;
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = mlpl_repl_connect::connect::create_session(&client, &base).unwrap();

        let mut frames: Vec<(String, usize, f64)> = Vec::new();
        let resp = mlpl_repl_connect::connect_stream::eval_remote_stream(
            &client,
            &base,
            &id,
            &token,
            "train 5 { loss_metric = step + 0.5 }",
            &mut |m| frames.push((m.name.clone(), m.step, m.value)),
        )
        .unwrap();

        assert_eq!(
            frames.len(),
            5,
            "expected 5 metric frames, got {} ({frames:?})",
            frames.len()
        );
        for (i, (name, step, value)) in frames.iter().enumerate() {
            assert_eq!(name, "loss_metric", "frame {i}: wrong name");
            assert_eq!(*step, i, "frame {i}: wrong step");
            let expected = i as f64 + 0.5;
            assert!(
                (*value - expected).abs() < 1e-9,
                "frame {i}: value {value} != {expected}"
            );
        }
        // Final value is the last `step_val` from the
        // train body -- here `step + 0.5` at step=4, so
        // 4.5. The server tags every `DenseArray` (even
        // rank-0 scalars) as `kind = "array"`.
        assert_eq!(resp.kind, "array");
        assert!(
            resp.value.contains("4.5"),
            "final value should be 4.5: {}",
            resp.value
        );
    })
    .await;
    result.unwrap();
}

/// `--stream` without `--connect` is a programmer
/// error: the streaming endpoint only makes sense in
/// remote mode. The parser returns `StreamWithoutConnect`
/// (the binary's wrapper translates that into exit
/// code 2 with a usage hint).
#[test]
fn parse_stream_without_connect_errors() {
    let args = vec!["mlpl-repl".to_string(), "--stream".to_string()];
    let err = mlpl_repl_connect::connect_stream::parse_connect_args(&args, None).unwrap_err();
    assert!(
        matches!(
            err,
            mlpl_repl_connect::connect_stream::ConnectArgsError::StreamWithoutConnect
        ),
        "expected StreamWithoutConnect, got {err:?}"
    );
}

/// `MLPL_REPL_STREAM=1` is equivalent to passing
/// `--stream` on the command line. The env path is the
/// muscle-memory ergonomic for users who always want
/// live loss.
#[test]
fn parse_stream_env_var_equivalent_to_flag() {
    let with_env = mlpl_repl_connect::connect_stream::parse_connect_args(
        &[
            "mlpl-repl".to_string(),
            "--connect".to_string(),
            "http://localhost:6464".to_string(),
        ],
        Some("1"),
    )
    .unwrap();
    let with_flag = mlpl_repl_connect::connect_stream::parse_connect_args(
        &[
            "mlpl-repl".to_string(),
            "--connect".to_string(),
            "http://localhost:6464".to_string(),
            "--stream".to_string(),
        ],
        None,
    )
    .unwrap();

    match (with_env, with_flag) {
        (
            mlpl_repl_connect::connect_stream::ConnectMode::Remote {
                url: u1,
                stream: s1,
                ..
            },
            mlpl_repl_connect::connect_stream::ConnectMode::Remote {
                url: u2,
                stream: s2,
                ..
            },
        ) => {
            assert!(s1, "env-var path should enable streaming");
            assert!(s2, "flag path should enable streaming");
            assert_eq!(u1, u2);
        }
        other => panic!("expected both Remote, got {other:?}"),
    }
}

/// `MLPL_REPL_STREAM=0` (or empty) is treated as
/// "streaming off" -- otherwise existing shells with the
/// var set to a falsy value would suddenly start
/// streaming everything.
#[test]
fn parse_stream_env_var_falsy_values_keep_streaming_off() {
    for falsy in ["0", "", "false", "FALSE", "no"] {
        let mode = mlpl_repl_connect::connect_stream::parse_connect_args(
            &[
                "mlpl-repl".to_string(),
                "--connect".to_string(),
                "http://localhost:6464".to_string(),
            ],
            Some(falsy),
        )
        .unwrap();
        match mode {
            mlpl_repl_connect::connect_stream::ConnectMode::Remote { stream, .. } => {
                assert!(!stream, "{falsy:?} should not enable streaming");
            }
            other => panic!("expected Remote, got {other:?}"),
        }
    }
}

/// No `--connect` and no `--stream` -> the wrapper
/// falls through to local mode (`Local` variant).
#[test]
fn parse_no_connect_no_stream_is_local() {
    let mode =
        mlpl_repl_connect::connect_stream::parse_connect_args(&["mlpl-repl".to_string()], None)
            .unwrap();
    assert!(matches!(
        mode,
        mlpl_repl_connect::connect_stream::ConnectMode::Local
    ));
}

/// A `metric` SSE frame renders as one progress line
/// containing the metric name, step index, and value.
/// The exact format is the live-loss display the REPL
/// shows during a `train { }`. We assert on substrings
/// so a future format tweak (color, padding) does not
/// require a test update.
#[test]
fn render_metric_writes_one_progress_line() {
    let mut buf: Vec<u8> = Vec::new();
    let m = mlpl_repl_connect::connect_stream::SseMetric {
        name: "loss_metric".into(),
        step: 3,
        value: 0.42,
    };
    mlpl_repl_connect::connect_stream::render_metric(&mut buf, &m).unwrap();
    let rendered = String::from_utf8(buf).unwrap();
    assert!(
        rendered.contains("loss_metric"),
        "expected 'loss_metric' in render: {rendered:?}"
    );
    assert!(
        rendered.contains("3"),
        "expected step '3' in render: {rendered:?}"
    );
    assert!(
        rendered.contains("0.42") || rendered.contains("0.420"),
        "expected value '0.42' in render: {rendered:?}"
    );
}
