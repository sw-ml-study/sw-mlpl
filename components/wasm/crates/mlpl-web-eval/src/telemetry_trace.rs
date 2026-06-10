//! Shared, persistent telemetry trace for connect-mode evals.
//!
//! The live `<TelemetryPanel>` writes each `/v1/stats` sample here (and
//! `reset()`s when a new eval begins). When the eval finishes, the
//! result entry appends `summary()` so the CPU/GPU/RAM/VRAM trace
//! PERSISTS in the REPL output instead of vanishing with the
//! "evaluating..." marker -- letting a brief GPU blip be seen after the
//! fact. `watch()` drives an on-demand bounded burst for `:status watch`.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use mlpl_monitor_types::Snapshot;
use mlpl_monitor_types::spark::{metric_percents, sparkline};

use crate::eval::ResultCb;

const WINDOW: usize = 60;
const LABELS: [&str; 4] = ["CPU ", "RAM ", "GPU ", "VRAM"];

thread_local! {
    static TRACE: RefCell<[Vec<u32>; 4]> =
        const { RefCell::new([Vec::new(), Vec::new(), Vec::new(), Vec::new()]) };
    // Whether the CURRENT eval runs on the server (so its backend
    // telemetry is meaningful). False for browser-local evals -- the
    // panel must not show server CPU for a computation happening in the
    // browser.
    static REMOTE: RefCell<bool> = const { RefCell::new(false) };
}

/// Begin a fresh trace for a new eval. `remote` records whether this
/// eval runs on the server; the live panel only shows for remote evals.
pub fn begin(remote: bool) {
    TRACE.with(|t| {
        for buf in t.borrow_mut().iter_mut() {
            buf.clear();
        }
    });
    REMOTE.with(|r| *r.borrow_mut() = remote);
}

/// Whether the current eval is server-side (panel-relevant).
#[must_use]
pub fn is_remote() -> bool {
    REMOTE.with(|r| *r.borrow())
}

/// Record one sample's four metric percentages.
pub fn push(s: &Snapshot) {
    let p = metric_percents(s);
    TRACE.with(|t| {
        let mut b = t.borrow_mut();
        for (buf, &v) in b.iter_mut().zip(p.iter()) {
            buf.push(v);
            if buf.len() > WINDOW {
                buf.remove(0);
            }
        }
    });
}

/// A compact 2-line sparkline summary with peaks, or `None` if no
/// samples were collected (e.g. the eval finished before the first
/// poll).
#[must_use]
pub fn summary() -> Option<String> {
    TRACE.with(|t| {
        let b = t.borrow();
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
    })
}

/// On-demand bounded burst for `:status watch`: reset, poll `/v1/stats`
/// `samples` times (~300ms apart), then fire `on_result` with the
/// sparkline summary. Auto-stops -- no infinite loop, no exit needed.
pub fn watch(base_url: String, samples: u32, on_result: ResultCb) {
    begin(true);
    wasm_bindgen_futures::spawn_local(async move {
        let url = format!("{}/v1/stats", base_url.trim_end_matches('/'));
        for _ in 0..samples {
            // Bound EACH sample so a slow /v1/stats (e.g. while a big
            // Ollama model loads and nvidia-smi blocks) can't hang the
            // whole watch loop. A timed-out sample is skipped.
            let got = crate::eval_wasm_helpers::with_deadline(1500, "stats", async {
                let resp = gloo::net::http::Request::get(&url)
                    .send()
                    .await
                    .map_err(|e| e.to_string())?;
                resp.json::<Snapshot>().await.map_err(|e| e.to_string())
            })
            .await;
            if let Ok(snap) = got {
                push(&snap);
            }
            gloo::timers::future::TimeoutFuture::new(300).await;
        }
        on_result(summary().unwrap_or_else(|| {
            "watch: backend did not respond -- is it up? (check :status)".to_string()
        }));
    });
}
