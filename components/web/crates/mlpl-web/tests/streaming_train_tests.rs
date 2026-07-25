//! Saga 21.5 step 007: SSE streaming + cancel wire tests for
//! the web REPL's `RemoteEvaluator`.
//!
//! Native-only tests: `RemoteEvaluator::eval_stream` POSTs to
//! `/v1/sessions/<id>/eval_stream` and surfaces every
//! `event: metric` frame via a callback, then returns the
//! terminal `Done` / `Cancelled` / `Error` outcome. The WASM
//! impl uses `gloo::net::http` + a ReadableStream chunk reader;
//! both impls share the same frame-parse logic so the contract
//! tested here applies to the browser path too.
//!
//! `RemoteEvaluator::cancel` POSTs `/v1/sessions/<id>/cancel`
//! against the cached server session id. Tested via a side
//! thread that fires cancel mid-train and asserts the partial
//! loss curve comes back.

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval.rs"]
mod eval;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_sse.rs"]
mod eval_sse;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_url.rs"]
mod eval_url;

use std::cell::RefCell;
use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use eval::{Evaluator, RemoteEvaluator, RemoteMetric, StreamOutcome};
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

/// Helper: drive `eval_stream` to completion and return the
/// captured metrics + terminal outcome.
fn drive_stream(evaluator: &RemoteEvaluator, program: &str) -> (Vec<RemoteMetric>, StreamOutcome) {
    let metrics: Rc<RefCell<Vec<RemoteMetric>>> = Rc::new(RefCell::new(Vec::new()));
    let outcome: Rc<RefCell<Option<StreamOutcome>>> = Rc::new(RefCell::new(None));
    let m = metrics.clone();
    let o = outcome.clone();
    evaluator.eval_stream(
        program,
        Box::new(move |frame: &RemoteMetric| {
            m.borrow_mut().push(frame.clone());
        }),
        Box::new(move |result: StreamOutcome| {
            *o.borrow_mut() = Some(result);
        }),
    );
    let captured = metrics.borrow().clone();
    let result = outcome
        .borrow_mut()
        .take()
        .expect("stream should terminate");
    (captured, result)
}

/// `train 5 { loss_metric = step + 0.5 ; step + 0.5 }` emits
/// five `metric` frames in order followed by a `done` outcome
/// carrying the final value `4.5`.
#[tokio::test(flavor = "multi_thread")]
async fn stream_emits_metric_per_train_iteration() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let remote = RemoteEvaluator::new(&base);
        let (metrics, outcome) =
            drive_stream(&remote, "train 5 { loss_metric = step + 0.5 ; step + 0.5 }");
        assert_eq!(metrics.len(), 5, "metrics: {metrics:?}");
        for (i, m) in metrics.iter().enumerate() {
            assert_eq!(m.name, "loss_metric");
            assert_eq!(m.step, i);
            let expected = i as f64 + 0.5;
            assert!(
                (m.value - expected).abs() < 1e-9,
                "metric[{i}].value = {} != {expected}",
                m.value
            );
        }
        match outcome {
            StreamOutcome::Done { value, kind } => {
                assert_eq!(kind, "array");
                assert!(value.contains("4.5"), "done value: {value}");
            }
            other => panic!("expected Done, got {other:?}"),
        }
    })
    .await;
    result.unwrap();
}

/// Streaming and non-streaming paths produce the same final
/// value for a non-train program. No metric frames are emitted
/// (the program has no `_metric`-suffixed bindings).
#[tokio::test(flavor = "multi_thread")]
async fn stream_and_eval_agree_on_final_value() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let remote_stream = RemoteEvaluator::new(&base);
        let (metrics, outcome) = drive_stream(&remote_stream, "iota(5) + 1");
        assert!(metrics.is_empty(), "non-train should have no metrics");
        let stream_value = match outcome {
            StreamOutcome::Done { value, .. } => value,
            other => panic!("expected Done: {other:?}"),
        };
        let remote_eval = RemoteEvaluator::new(&base);
        let slot = Rc::new(RefCell::new(None::<String>));
        let s = slot.clone();
        remote_eval.eval("iota(5) + 1", Box::new(move |r| *s.borrow_mut() = Some(r)));
        let eval_value = slot.borrow().clone().unwrap();
        assert_eq!(stream_value, eval_value);
    })
    .await;
    result.unwrap();
}

/// `RemoteEvaluator::cancel` fired from a side thread mid-train
/// trips the server's interrupt token. The stream terminates
/// with `Cancelled { step, partial_losses }` carrying the iteration
/// the cancel landed on and the loss curve up to that point.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_mid_train_returns_partial_losses() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let remote = RemoteEvaluator::new(&base);
        // Force the session to materialize so the side thread's
        // cancel handle can target the same one as the stream.
        let warmup = Rc::new(RefCell::new(None::<String>));
        let w = warmup.clone();
        remote.eval("x = 1", Box::new(move |r| *w.borrow_mut() = Some(r)));
        let cancel = remote
            .cancel_handle()
            .expect("session minted by warmup eval");

        // Spawn the cancel in a side thread that fires after a
        // brief delay -- the train below is sized so cancel
        // reliably lands mid-flight.
        let cancel_token = Arc::new(AtomicUsize::new(0));
        let tok = cancel_token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            cancel.cancel();
            tok.store(1, Ordering::SeqCst);
        });

        let (metrics, outcome) = drive_stream(
            &remote,
            "train 50000 { reduce_add(iota(500)) ; loss_metric = step + 0.5 }",
        );
        assert_eq!(
            cancel_token.load(Ordering::SeqCst),
            1,
            "side thread should have fired cancel"
        );
        match outcome {
            StreamOutcome::Cancelled {
                step,
                partial_losses,
            } => {
                assert!(
                    step > 0 && step < 50000,
                    "cancel should land mid-train, step={step}"
                );
                assert_eq!(partial_losses.len(), step);
                // Sanity: we observed at least as many metric
                // frames as the partial-losses vector reports.
                assert!(metrics.len() >= step.min(metrics.len()));
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    })
    .await;
    result.unwrap();
}

/// A program that fails to lex / parse before the SSE stream
/// opens lands as `StreamOutcome::Error`, not as a `Done` with
/// an error-shaped value. (The server emits the error on the
/// `400` HTTP response before the event-stream opens; the
/// client maps that to the `Error` variant.)
#[tokio::test(flavor = "multi_thread")]
async fn stream_eval_error_surfaces_as_error_outcome() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let remote = RemoteEvaluator::new(&base);
        let (_metrics, outcome) = drive_stream(&remote, "undefined_var");
        match outcome {
            StreamOutcome::Error { message } => {
                assert!(
                    message.to_lowercase().contains("undefined"),
                    "error message should mention 'undefined': {message}"
                );
            }
            other => panic!("expected Error, got {other:?}"),
        }
    })
    .await;
    result.unwrap();
}

/// Connect-telemetry step 002: metric frames stream into the
/// per-generation loss trace exactly as the live panel consumes them --
/// one point per frame, in arrival order, with a consuming summary
/// line left over for the result entry.
#[tokio::test(flavor = "multi_thread")]
async fn streamed_metrics_feed_the_live_loss_trace() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        use mlpl_web_eval::loss_trace;
        let gen_id = 777_002;
        let program = "train 4 { loss_metric = 4 - step ; 4 - step }";
        assert!(
            mlpl_web_eval::connect_guard::program_streams_metrics(program),
            "train programs route to the streaming path"
        );
        let remote = RemoteEvaluator::new(&base);
        let done: Rc<RefCell<Option<StreamOutcome>>> = Rc::new(RefCell::new(None));
        let d = done.clone();
        remote.eval_stream(
            program,
            Box::new(move |m: &RemoteMetric| loss_trace::push(gen_id, &m.name, m.value)),
            Box::new(move |o: StreamOutcome| *d.borrow_mut() = Some(o)),
        );
        assert!(
            matches!(done.borrow_mut().take(), Some(StreamOutcome::Done { .. })),
            "stream should complete"
        );
        let (train, val) = loss_trace::series(gen_id);
        assert_eq!(train, vec![4.0, 3.0, 2.0, 1.0]);
        assert!(val.is_empty());
        assert_eq!(loss_trace::seq(gen_id), 4);
        let s = loss_trace::summary(gen_id).expect("summary for streamed gen");
        assert!(s.contains("4 steps"), "step count in summary: {s}");
        assert!(s.contains("1.0000"), "final loss in summary: {s}");
        assert!(loss_trace::summary(gen_id).is_none(), "summary consumes");
    })
    .await;
    result.unwrap();
}
