//! Connect-telemetry step 002: per-eval live-loss series store.
//!
//! `loss_trace` mirrors `telemetry_trace`'s generation model but holds
//! streamed `*_metric` training series instead of hardware samples, and
//! is NOT wasm-gated so the reducer logic tests natively.

use mlpl_web_eval::connect_guard::program_streams_metrics;
use mlpl_web_eval::loss_trace;

// Generations in these tests are arbitrary but unique per test so the
// shared thread-local store never cross-talks (tests may share a thread).

#[test]
fn push_builds_a_train_series_in_arrival_order() {
    let gen_id = 9001;
    loss_trace::push(gen_id, "loss_metric", 3.0);
    loss_trace::push(gen_id, "loss_metric", 2.0);
    loss_trace::push(gen_id, "loss_metric", 1.0);
    let (train, val) = loss_trace::series(gen_id);
    assert_eq!(train, vec![3.0, 2.0, 1.0]);
    assert!(val.is_empty());
}

#[test]
fn val_prefixed_metric_lands_in_the_val_series() {
    let gen_id = 9002;
    loss_trace::push(gen_id, "loss_metric", 3.0);
    loss_trace::push(gen_id, "val_loss_metric", 4.0);
    loss_trace::push(gen_id, "loss_metric", 2.0);
    loss_trace::push(gen_id, "val_loss_metric", 3.5);
    let (train, val) = loss_trace::series(gen_id);
    assert_eq!(train, vec![3.0, 2.0]);
    assert_eq!(val, vec![4.0, 3.5]);
}

#[test]
fn non_loss_metric_still_charts_as_the_primary_series() {
    let gen_id = 9003;
    loss_trace::push(gen_id, "acc_metric", 0.5);
    loss_trace::push(gen_id, "acc_metric", 0.75);
    let (train, val) = loss_trace::series(gen_id);
    assert_eq!(train, vec![0.5, 0.75]);
    assert!(val.is_empty());
}

#[test]
fn generations_stay_isolated() {
    loss_trace::push(9004, "loss_metric", 1.0);
    loss_trace::push(9005, "loss_metric", 9.0);
    let (a, _) = loss_trace::series(9004);
    let (b, _) = loss_trace::series(9005);
    assert_eq!(a, vec![1.0]);
    assert_eq!(b, vec![9.0]);
}

#[test]
fn seq_increments_per_push_and_reads_zero_for_unknown() {
    let gen_id = 9006;
    assert_eq!(loss_trace::seq(gen_id), 0);
    loss_trace::push(gen_id, "loss_metric", 1.0);
    loss_trace::push(gen_id, "val_loss_metric", 2.0);
    assert_eq!(loss_trace::seq(gen_id), 2);
}

#[test]
fn summary_consumes_and_names_final_loss() {
    let gen_id = 9007;
    loss_trace::push(gen_id, "loss_metric", 4.0);
    loss_trace::push(gen_id, "loss_metric", 0.125);
    let s = loss_trace::summary(gen_id).expect("summary for populated gen");
    assert!(s.contains("loss"), "summary names the series: {s}");
    assert!(s.contains("0.125"), "summary shows final value: {s}");
    assert!(s.contains('2'), "summary shows step count: {s}");
    assert!(loss_trace::summary(gen_id).is_none(), "summary consumes");
}

#[test]
fn summary_is_none_for_empty_or_unknown_gen() {
    assert!(loss_trace::summary(424_242).is_none());
}

#[test]
fn train_programs_stream_metrics_other_programs_do_not() {
    assert!(program_streams_metrics(
        "train 5 { loss_metric = step ; step }"
    ));
    assert!(program_streams_metrics(
        "x = 1 ; train 30 {\n  loss_metric = l\n}"
    ));
    assert!(!program_streams_metrics("x = range(5) + 1"));
    assert!(!program_streams_metrics(":ask how do I train?"));
    // The `:ask` shortcut becomes an llm_call program -- the word
    // "train" inside the question must not route it to the stream path.
    assert!(!program_streams_metrics(
        "llm_call(\"http://x\", \"how does train 5 { } work?\", \"m\", \"s\")"
    ));
    // "train" as a substring of an identifier is not a train block.
    assert!(!program_streams_metrics("retrain_flag = 1"));
    assert!(!program_streams_metrics("trainx = { }"));
}

#[test]
fn spark_line_normalizes_to_glyph_range() {
    let s = loss_trace::spark_line(&[4.0, 3.0, 2.0, 1.0]);
    assert_eq!(s.chars().count(), 4);
    let first = s.chars().next().unwrap();
    let last = s.chars().last().unwrap();
    assert!(first > last, "falling loss renders falling bars: {s}");
    assert!(loss_trace::spark_line(&[]).is_empty());
}
