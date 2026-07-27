//! Per-eval live loss series streamed from `event: metric` SSE frames.
//!
//! Generation-keyed twin of `telemetry_trace`, but pure std (no wasm
//! gate) so the reducer logic tests natively: the WASM connect path
//! pushes one sample per metric frame, the `LiveLossPanel` polls
//! `seq`/`series` to repaint, and the result callback `summary`s the
//! generation into a persistent one-liner when the eval finishes.

use std::cell::RefCell;
use std::collections::HashMap;

const KEEP: u32 = 6;
const CAP: usize = 4096;

/// Streamed series keyed by metric name in first-seen order, plus a
/// monotonic push counter the panel polls for repaint.
#[derive(Default)]
struct Trace {
    series: Vec<(String, Vec<f64>)>,
    seq: u32,
}

thread_local! {
    static TRACES: RefCell<HashMap<u32, Trace>> = RefCell::new(HashMap::new());
}

/// Append one streamed metric sample to generation `gen_id`. Series are
/// capped at `CAP` points (train demos stay far below); generations
/// more than `KEEP` behind this one are pruned, mirroring
/// `telemetry_trace`.
pub fn push(gen_id: u32, name: &str, value: f64) {
    TRACES.with(|t| {
        let mut m = t.borrow_mut();
        m.retain(|&k, _| gen_id.saturating_sub(KEEP) < k);
        let trace = m.entry(gen_id).or_default();
        let idx = match trace.series.iter().position(|(n, _)| n == name) {
            Some(i) => i,
            None => {
                trace.series.push((name.to_string(), Vec::new()));
                trace.series.len() - 1
            }
        };
        let buf = &mut trace.series[idx].1;
        if buf.len() < CAP {
            buf.push(value);
        }
        trace.seq = trace.seq.wrapping_add(1);
    });
}

/// Monotonic per-generation push counter (0 for unknown generations).
#[must_use]
pub fn seq(gen_id: u32) -> u32 {
    TRACES.with(|t| t.borrow().get(&gen_id).map_or(0, |tr| tr.seq))
}

/// The `(train, val)` series for `gen_id`, empty when unknown. Val is
/// the first series whose name starts with `val`; train is the first
/// non-val series, preferring a name containing `loss` (so a stray
/// `acc_metric` still charts when it is all the program emits).
#[must_use]
pub fn series(gen_id: u32) -> (Vec<f64>, Vec<f64>) {
    TRACES.with(|t| {
        t.borrow().get(&gen_id).map_or_else(
            || (Vec::new(), Vec::new()),
            |tr| {
                let (train, val) = picks(&tr.series);
                (
                    train.cloned().unwrap_or_default(),
                    val.cloned().unwrap_or_default(),
                )
            },
        )
    })
}

/// `(train, val)` series selection shared by `series` and `summary`.
fn picks(series: &[(String, Vec<f64>)]) -> (Option<&Vec<f64>>, Option<&Vec<f64>>) {
    let non_val = |n: &str| !n.starts_with("val");
    let train = series
        .iter()
        .find(|(n, _)| non_val(n) && n.contains("loss"))
        .or_else(|| series.iter().find(|(n, _)| non_val(n)))
        .map(|(_, b)| b);
    let val = series
        .iter()
        .find(|(n, _)| n.starts_with("val"))
        .map(|(_, b)| b);
    (train, val)
}

/// One-line text record of the streamed loss (sparkline + step count +
/// final values), CONSUMING the generation -- appended under the result
/// so the live curve leaves a durable trace once the panel unmounts.
/// `None` when nothing was streamed.
#[must_use]
pub fn summary(gen_id: u32) -> Option<String> {
    let trace = TRACES.with(|t| t.borrow_mut().remove(&gen_id))?;
    let (train, val) = picks(&trace.series);
    let train = train?;
    let last = *train.last()?;
    let spark = mlpl_monitor_types::spark::sparkline(&percents(train), 100);
    let val_note = val
        .and_then(|v| v.last())
        .map(|v| format!("  val {v:.4}"))
        .unwrap_or_default();
    Some(format!(
        "live loss {spark}  {} steps, final {last:.4}{val_note}",
        train.len()
    ))
}

/// Normalize a series into 0..=100 integer percentages for the text
/// sparkline (min -> 0, max -> 100; a constant series reads as flat
/// 50s). The clamp bounds the value, so the cast cannot truncate.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn percents(vals: &[f64]) -> Vec<u32> {
    let (min, max) = vals
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let range = max - min;
    vals.iter()
        .map(|&v| {
            if range > 0.0 {
                ((v - min) / range * 100.0).clamp(0.0, 100.0).round() as u32
            } else {
                50
            }
        })
        .collect()
}

/// Text sparkline of a loss series (min -> lowest bar, max -> highest),
/// for the LOSS row the telemetry panel time-aligns with its hardware
/// rows. Empty input yields an empty string.
#[must_use]
pub fn spark_line(vals: &[f64]) -> String {
    mlpl_monitor_types::spark::sparkline(&percents(vals), 100)
}
