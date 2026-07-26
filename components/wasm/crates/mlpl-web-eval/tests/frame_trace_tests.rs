//! `event: frame` SSE payloads land in the generation-keyed
//! `frame_trace` store (Game of Life saga step 4), surviving
//! arbitrary chunk boundaries like every other SSE frame.

use mlpl_web_eval::eval::{RemoteMetric, StreamOutcome};
use mlpl_web_eval::eval_sse::SseFeed;
use mlpl_web_eval::frame_trace;

fn feed_all(feed: &mut SseFeed, body: &str) -> Option<StreamOutcome> {
    let mut cb: Box<dyn FnMut(&RemoteMetric)> = Box::new(|_| {});
    feed.push(body.as_bytes(), &mut cb)
}

#[test]
fn frame_event_stores_latest_board() {
    let mut feed = SseFeed::default().with_generation(901);
    let body = "event: frame\ndata: {\"name\":\"life\",\"step\":0,\"shape\":[2,2],\"values\":[1.0,0.0,0.0,1.0]}\n\n\
event: frame\ndata: {\"name\":\"life\",\"step\":1,\"shape\":[2,2],\"values\":[0.0,1.0,1.0,0.0]}\n\n\
event: done\ndata: {\"value\":\"ok\",\"kind\":\"array\"}\n\n";
    let outcome = feed_all(&mut feed, body);
    assert!(matches!(outcome, Some(StreamOutcome::Done { .. })));
    assert_eq!(frame_trace::seq(901), 2);
    let (name, step, shape, values) = frame_trace::latest(901).expect("latest frame");
    assert_eq!(name, "life");
    assert_eq!(step, 1);
    assert_eq!(shape, vec![2, 2]);
    assert_eq!(values, vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn frame_split_across_chunks_still_lands() {
    let mut feed = SseFeed::default().with_generation(902);
    let whole = "event: frame\ndata: {\"name\":\"life\",\"step\":4,\"shape\":[1,3],\"values\":[1.0,1.0,1.0]}\n\n";
    let (a, b) = whole.split_at(37);
    let mut cb: Box<dyn FnMut(&RemoteMetric)> = Box::new(|_| {});
    assert!(feed.push(a.as_bytes(), &mut cb).is_none());
    assert!(feed.push(b.as_bytes(), &mut cb).is_none());
    assert_eq!(frame_trace::seq(902), 1);
    assert_eq!(frame_trace::latest(902).unwrap().1, 4);
}

#[test]
fn old_generations_are_pruned() {
    frame_trace::push(100, "life", 0, &[1], &[1.0]);
    frame_trace::push(120, "life", 0, &[1], &[1.0]);
    assert!(
        frame_trace::latest(100).is_none(),
        "gen 100 pruned by gen 120"
    );
    assert!(frame_trace::latest(120).is_some());
}
