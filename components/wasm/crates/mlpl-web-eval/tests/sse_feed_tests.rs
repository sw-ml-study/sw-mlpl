//! Connect-telemetry step 002: incremental SSE feed parser.
//!
//! `SseFeed` is the push-based twin of `parse_sse_stream`: browser
//! `ReadableStream` chunks arrive at arbitrary byte boundaries (mid-line,
//! mid-frame, even mid-UTF-8), and the feed must assemble exactly the
//! same metric callbacks + terminal outcome as the pull parser gets
//! from a complete body.

use mlpl_web_eval::eval::{MetricCb, RemoteMetric, StreamOutcome};
use mlpl_web_eval::eval_sse::SseFeed;

const BODY: &str = "event: ready\ndata: {}\n\n\
    event: metric\ndata: {\"name\":\"loss_metric\",\"step\":0,\"value\":0.5}\n\n\
    event: metric\ndata: {\"name\":\"loss_metric\",\"step\":1,\"value\":0.25}\n\n\
    event: done\ndata: {\"value\":\"0.25\",\"kind\":\"scalar\"}\n\n";

fn collect(chunks: &[&[u8]]) -> (Vec<RemoteMetric>, Option<StreamOutcome>) {
    let metrics = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = metrics.clone();
    let mut cb: MetricCb = Box::new(move |m| sink.borrow_mut().push(m.clone()));
    let mut feed = SseFeed::default();
    let mut outcome = None;
    for chunk in chunks {
        if let Some(o) = feed.push(chunk, &mut cb) {
            outcome = Some(o);
            break;
        }
    }
    let got = metrics.borrow().clone();
    (got, outcome)
}

#[test]
fn whole_body_in_one_chunk_matches_pull_parser() {
    let (metrics, outcome) = collect(&[BODY.as_bytes()]);
    assert_eq!(metrics.len(), 2);
    assert_eq!(metrics[0].name, "loss_metric");
    assert_eq!(metrics[0].step, 0);
    assert!((metrics[0].value - 0.5).abs() < 1e-12);
    assert!((metrics[1].value - 0.25).abs() < 1e-12);
    match outcome {
        Some(StreamOutcome::Done { value, kind }) => {
            assert_eq!(value, "0.25");
            assert_eq!(kind, "scalar");
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[test]
fn byte_at_a_time_chunks_assemble_identically() {
    let bytes = BODY.as_bytes();
    let chunks: Vec<&[u8]> = bytes.chunks(1).collect();
    let (metrics, outcome) = collect(&chunks);
    assert_eq!(metrics.len(), 2);
    assert!(matches!(outcome, Some(StreamOutcome::Done { .. })));
}

#[test]
fn split_mid_frame_and_mid_json_is_reassembled() {
    // Split inside "metric" event name and inside the JSON payload.
    let parts: Vec<&[u8]> = vec![
        b"event: met",
        b"ric\ndata: {\"name\":\"loss_",
        b"metric\",\"step\":0,\"va",
        b"lue\":1.5}\n\nevent: done\ndata: {\"value\":\"x\",\"kind\":\"scalar\"}\n\n",
    ];
    let (metrics, outcome) = collect(&parts);
    assert_eq!(metrics.len(), 1);
    assert!((metrics[0].value - 1.5).abs() < 1e-12);
    assert!(matches!(outcome, Some(StreamOutcome::Done { .. })));
}

#[test]
fn crlf_line_endings_are_accepted() {
    let body = BODY.replace('\n', "\r\n");
    let (metrics, outcome) = collect(&[body.as_bytes()]);
    assert_eq!(metrics.len(), 2);
    assert!(matches!(outcome, Some(StreamOutcome::Done { .. })));
}

#[test]
fn error_frame_terminates_with_error_outcome() {
    let body = "event: error\ndata: {\"error\":\"boom\"}\n\n";
    let (metrics, outcome) = collect(&[body.as_bytes()]);
    assert!(metrics.is_empty());
    match outcome {
        Some(StreamOutcome::Error { message }) => assert_eq!(message, "boom"),
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn cancelled_frame_carries_partial_losses() {
    let body = "event: cancelled\ndata: {\"step\":3,\"partial_losses\":[3.0,2.0,1.0]}\n\n";
    let (_, outcome) = collect(&[body.as_bytes()]);
    match outcome {
        Some(StreamOutcome::Cancelled {
            step,
            partial_losses,
        }) => {
            assert_eq!(step, 3);
            assert_eq!(partial_losses, vec![3.0, 2.0, 1.0]);
        }
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[test]
fn into_display_maps_outcomes_to_repl_strings() {
    let (done, err) = StreamOutcome::Done {
        value: "42".into(),
        kind: "scalar".into(),
    }
    .into_display();
    assert_eq!(done, "42");
    assert!(!err);
    let (cancelled, err) = StreamOutcome::Cancelled {
        step: 7,
        partial_losses: vec![1.0; 7],
    }
    .into_display();
    assert!(cancelled.contains("cancelled"));
    assert!(cancelled.contains('7'));
    assert!(!err);
    let (msg, err) = StreamOutcome::Error {
        message: "nope".into(),
    }
    .into_display();
    assert_eq!(msg, "error: nope");
    assert!(err);
}
