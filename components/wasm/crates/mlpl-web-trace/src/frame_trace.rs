//! Latest streamed tensor FRAME per eval generation (Game of Life
//! saga step 4) -- the whole-board twin of `loss_trace`. `event:
//! frame` SSE frames land here via `SseFeed`; the `LiveLifePanel`
//! polls `seq`/`latest` on its own clock and repaints the grid.
//! Pure std (no wasm gate) so the reducer logic tests natively.
//! Only the LATEST frame per generation is kept: the live view
//! shows "now"; the durable animation is the SMIL value the
//! program returns at the end.

use std::cell::RefCell;
use std::collections::HashMap;

const KEEP: u32 = 6;

/// Latest frame + a monotonic push counter for repaint polling.
#[derive(Default, Clone)]
struct Latest {
    name: String,
    step: usize,
    shape: Vec<usize>,
    values: Vec<f64>,
    seq: u32,
}

thread_local! {
    static FRAMES: RefCell<HashMap<u32, Latest>> = RefCell::new(HashMap::new());
}

/// Record generation `gen_id`'s newest frame, pruning generations
/// more than `KEEP` behind (mirrors `loss_trace`).
pub fn push(gen_id: u32, name: &str, step: usize, shape: &[usize], values: &[f64]) {
    FRAMES.with(|f| {
        let mut m = f.borrow_mut();
        m.retain(|&k, _| gen_id.saturating_sub(KEEP) < k);
        let entry = m.entry(gen_id).or_default();
        entry.name = name.to_string();
        entry.step = step;
        entry.shape = shape.to_vec();
        entry.values = values.to_vec();
        entry.seq = entry.seq.wrapping_add(1);
    });
}

/// Monotonic per-generation push counter (0 for unknown).
#[must_use]
pub fn seq(gen_id: u32) -> u32 {
    FRAMES.with(|f| f.borrow().get(&gen_id).map_or(0, |l| l.seq))
}

/// The newest `(name, step, shape, values)` for `gen_id`.
#[must_use]
pub fn latest(gen_id: u32) -> Option<(String, usize, Vec<usize>, Vec<f64>)> {
    FRAMES.with(|f| {
        f.borrow()
            .get(&gen_id)
            .map(|l| (l.name.clone(), l.step, l.shape.clone(), l.values.clone()))
    })
}

/// Store an `event: frame` JSON payload (from the `emit_frame`
/// builtin via the SSE wire) into `gen_id`'s slot. Malformed
/// payloads are dropped silently -- a bad frame must never kill
/// the stream that carries the eval's real result.
pub fn push_json(v: &serde_json::Value, gen_id: u32) {
    let (Some(name), Some(step), Some(shape), Some(values)) = (
        v.get("name").and_then(|x| x.as_str()),
        v.get("step").and_then(serde_json::Value::as_u64),
        v.get("shape").and_then(|x| x.as_array()),
        v.get("values").and_then(|x| x.as_array()),
    ) else {
        return;
    };
    let shape: Vec<usize> = shape
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .map(|d| d as usize)
        .collect();
    let values: Vec<f64> = values
        .iter()
        .filter_map(serde_json::Value::as_f64)
        .collect();
    push(gen_id, name, step as usize, &shape, &values);
}
