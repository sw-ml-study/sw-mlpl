//! Per-eval telemetry traces for connect-mode evals.
//!
//! Each eval gets its own GENERATION: `begin()` mints a new gen_id with
//! empty buffers; the live `<TelemetryPanel>` captures the gen_id at mount
//! and `push(gen_id, ..)`es each `/v1/stats` sample to it; when the eval
//! finishes the result entry reads `summary(gen_id)`. So a `:status watch`
//! and an `:ask` (or two concurrent server evals) never share buffers --
//! their sparklines stay isolated. Old gens are pruned (bounded memory);
//! `summary` removes the gen_id it reads.

#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use mlpl_monitor_types::Snapshot;
use mlpl_monitor_types::spark::{metric_percents, sparkline};

use crate::eval::ResultCb;

const WINDOW: usize = 60;
const KEEP: u32 = 6;
const LABELS: [&str; 4] = ["CPU ", "RAM ", "GPU ", "VRAM"];

thread_local! {
    static GEN: Cell<u32> = const { Cell::new(0) };
    static TRACES: RefCell<HashMap<u32, [Vec<u32>; 4]>> = RefCell::new(HashMap::new());
    static REMOTE: RefCell<HashMap<u32, bool>> = RefCell::new(HashMap::new());
}

/// Start a new eval generation. `remote` records whether this eval runs
/// on the server (the live panel only shows for remote gens). Returns
/// the gen_id; consumers more commonly read [`current_gen`] right after.
pub fn begin(remote: bool) -> u32 {
    let g = GEN.with(|c| {
        let n = c.get().wrapping_add(1);
        c.set(n);
        n
    });
    TRACES.with(|t| {
        let mut m = t.borrow_mut();
        m.insert(g, [Vec::new(), Vec::new(), Vec::new(), Vec::new()]);
        m.retain(|&k, _| g.saturating_sub(KEEP) < k);
    });
    REMOTE.with(|r| {
        let mut m = r.borrow_mut();
        m.insert(g, remote);
        m.retain(|&k, _| g.saturating_sub(KEEP) < k);
    });
    g
}

/// The current (latest) generation -- captured by the panel at mount and
/// by the result callback when it is created (both synchronous with the
/// `begin()` for their eval).
#[must_use]
pub fn current_gen() -> u32 {
    GEN.with(Cell::get)
}

/// Whether generation `gen_id` is a server-side (panel-relevant) eval.
#[must_use]
pub fn is_remote(gen_id: u32) -> bool {
    REMOTE.with(|r| r.borrow().get(&gen_id).copied().unwrap_or(false))
}

/// Record one sample into generation `gen_id` (no-op if pruned/unknown).
pub fn push(gen_id: u32, s: &Snapshot) {
    let p = metric_percents(s);
    TRACES.with(|t| {
        if let Some(b) = t.borrow_mut().get_mut(&gen_id) {
            for (buf, &v) in b.iter_mut().zip(p.iter()) {
                buf.push(v);
                if buf.len() > WINDOW {
                    buf.remove(0);
                }
            }
        }
    });
}

/// A compact 2-line sparkline summary (with peaks) for generation `gen_id`,
/// consuming it. `None` if it has no samples (or was pruned).
#[must_use]
pub fn summary(gen_id: u32) -> Option<String> {
    let b = TRACES.with(|t| t.borrow_mut().remove(&gen_id))?;
    if b[0].is_empty() {
        return None;
    }
    let row = |i: usize| {
        let peak = b[i].iter().copied().max().unwrap_or(0);
        format!("{}{} {peak:>3}% peak", LABELS[i], sparkline(&b[i], 100))
    };
    Some(format!(
        "backend load during eval:\n  {}   {}\n  {}   {}",
        row(0),
        row(2),
        row(1),
        row(3)
    ))
}

/// On-demand bounded burst for `:status watch`: its own generation, poll
/// `/v1/stats` `samples` times (~300ms apart, each bounded), then fire
/// `on_result` with the sparkline summary. Auto-stops.
pub fn watch(base_url: String, samples: u32, on_result: ResultCb) {
    let gen_id = begin(true);
    wasm_bindgen_futures::spawn_local(async move {
        let url = format!("{}/v1/stats", base_url.trim_end_matches('/'));
        for _ in 0..samples {
            let got = crate::eval_wasm_helpers::with_deadline(1500, "stats", async {
                let resp = gloo::net::http::Request::get(&url)
                    .send()
                    .await
                    .map_err(|e| e.to_string())?;
                resp.json::<Snapshot>().await.map_err(|e| e.to_string())
            })
            .await;
            if let Ok(snap) = got {
                push(gen_id, &snap);
            }
            gloo::timers::future::TimeoutFuture::new(300).await;
        }
        on_result(summary(gen_id).unwrap_or_else(|| {
            "watch: backend did not respond -- is it up? (check :status)".to_string()
        }));
    });
}
