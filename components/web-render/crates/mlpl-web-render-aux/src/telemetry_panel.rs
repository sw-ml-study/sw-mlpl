//! `<TelemetryPanel>` -- live backend CPU/GPU/RAM/VRAM sparklines shown
//! beside the "evaluating..." marker during a connect-mode eval.
//!
//! Mount starts a ~2s poll of `GET /v1/stats`; unmount (when the marker
//! is replaced by the result) drops the interval, so polling is bounded
//! to the lifetime of the running line. Renders nothing in local mode
//! (no connected backend). Sample buffers live in a `RefCell` (the poll
//! closure appends without the stale state-handle problem); a
//! seq-bumped `use_state` drives the re-render -- bumped on EVERY poll
//! (success or error) so a failed fetch surfaces its error rather than
//! freezing the panel.

use std::cell::RefCell;
use std::rc::Rc;

use gloo::timers::callback::Interval;
use mlpl_monitor_types::Snapshot;
use mlpl_monitor_types::spark::{metric_percents, sparkline};
use yew::prelude::*;

// Poll fast (the /v1/stats CPU window is ~120ms, so ~400ms avoids
// overlap) so even a sub-second GPU fine-tune yields several samples
// instead of a single bar.
const POLL_MS: u32 = 400;
const WINDOW: usize = 48;
const LABELS: [&str; 4] = ["CPU", "RAM", "GPU", "VRAM"];

/// Ring buffers of the four metric percentages, plus the freshest
/// snapshot, the last fetch error (shown when polling fails), and a
/// monotonic seq the component mirrors into a `use_state` to force a
/// re-render.
#[derive(Default)]
struct Series {
    rows: [Vec<u32>; 4],
    /// Streamed training loss sampled on the SAME poll clock as the
    /// hardware rows (connect-telemetry step 006), so "GPU busy" and
    /// "loss falling" read as one time-aligned story.
    loss: Vec<f64>,
    latest: Option<Snapshot>,
    note: Option<String>,
    seq: u32,
}

impl Series {
    fn push(&mut self, s: Snapshot) {
        let pcts = metric_percents(&s);
        for (buf, p) in self.rows.iter_mut().zip(pcts) {
            buf.push(p);
            if buf.len() > WINDOW {
                buf.remove(0);
            }
        }
        self.latest = Some(s);
        self.note = None;
        self.seq = self.seq.wrapping_add(1);
    }
}

/// `GET <url>` -> Snapshot, with a human-readable error string on any
/// transport / status / decode failure (surfaced in the panel).
async fn fetch_snapshot(url: &str) -> Result<Snapshot, String> {
    let resp = gloo::net::http::Request::get(url)
        .send()
        .await
        .map_err(|e| format!("net: {e}"))?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<Snapshot>()
        .await
        .map_err(|e| format!("decode: {e}"))
}

/// One poll: fetch, record success or error into the buffers, then bump
/// `tick` (its new value forces the component to re-render either way).
/// Successful snapshots also mirror into THIS eval's generation so the
/// sparkline PERSISTS in the result after the marker unmounts, isolated
/// from any other concurrent eval/watch.
fn poll_once(base: &str, gen_id: u32, series: &Rc<RefCell<Series>>, tick: &UseStateHandle<u32>) {
    let url = format!("{}/v1/stats", base.trim_end_matches('/'));
    let series = series.clone();
    let tick = tick.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let outcome = fetch_snapshot(&url).await;
        let seq = {
            let mut s = series.borrow_mut();
            match outcome {
                Ok(snap) => {
                    mlpl_web_eval::telemetry_trace::push(gen_id, &snap);
                    s.push(snap);
                }
                Err(e) => {
                    s.note = Some(e);
                    s.seq = s.seq.wrapping_add(1);
                }
            }
            s.seq
        };
        tick.set(seq);
    });
}

fn render_rows(series: &Series) -> Html {
    let pcts = series.latest.as_ref().map_or([0u32; 4], metric_percents);
    let metrics = series
        .rows
        .iter()
        .enumerate()
        .map(|(i, buf)| metric_row(i, buf, pcts[i]))
        .collect::<Html>();
    let note = series.note.as_ref().map_or_else(
        || html! {},
        |n| html! { <span class="telemetry-note">{ format!("(stats {n})") }</span> },
    );
    html! {
        <div class="telemetry-panel">
            <div class="telemetry-line">{ metrics }{ loss_row(&series.loss) }{ note }</div>
        </div>
    }
}

/// One labeled sparkline cell (label / bars / current percent).
fn metric_row(i: usize, buf: &[u32], pct: u32) -> Html {
    html! {
        <span class="telemetry-metric">
            <span class="telemetry-label">{ LABELS[i] }</span>
            <span class="telemetry-spark">{ sparkline(buf, 100) }</span>
            <span class="telemetry-val">{ format!("{pct}%") }</span>
        </span>
    }
}

/// The time-aligned LOSS row: latest streamed loss + its sparkline on
/// the shared clock/window. Empty until the eval streams metric frames.
fn loss_row(loss: &[f64]) -> Html {
    let Some(&last) = loss.last() else {
        return html! {};
    };
    html! {
        <span class="telemetry-metric">
            <span class="telemetry-label">{ "LOSS" }</span>
            <span class="telemetry-spark">{ mlpl_web_eval::loss_trace::spark_line(loss) }</span>
            <span class="telemetry-val">{ format!("{last:.4}") }</span>
        </span>
    }
}

#[function_component(TelemetryPanel)]
pub fn telemetry_panel() -> Html {
    // Read THIS eval's generation EACH render (not captured once at mount):
    // a loaded demo can reuse the same panel instance across lines, where a
    // captured gen_id would stay the first line's (browser-local, never
    // remote) and the panel would never activate for a later `:ask`. The
    // panel only shows for a server-side eval -- a browser-local eval would
    // otherwise display server CPU unrelated to where it runs.
    let gen_id = mlpl_web_eval::telemetry_trace::current_gen();
    let base = (*use_state(mlpl_web_eval::eval::current_connect_url_from_window)).clone();
    let series = use_mut_ref(Series::default);
    let tick = use_state(|| 0u32);
    let active = base.is_some() && mlpl_web_eval::telemetry_trace::is_remote(gen_id);
    {
        let series = series.clone();
        let tick = tick.clone();
        let base = base.clone();
        // Re-run when EITHER `active` or the generation changes, so a panel
        // reused across demo lines restarts polling for the new eval.
        use_effect_with((active, gen_id), move |&(active, gen_id)| {
            // Loss sampling shares poll_once's clock so the LOSS row
            // stays column-aligned with the hardware rows.
            let sample_loss = {
                let series = series.clone();
                move |gen_id: u32| {
                    if let Some(&v) = mlpl_web_eval::loss_trace::series(gen_id).0.last() {
                        let mut s = series.borrow_mut();
                        s.loss.push(v);
                        if s.loss.len() > WINDOW {
                            s.loss.remove(0);
                        }
                    }
                }
            };
            let mut interval = None;
            if let (true, Some(b)) = (active, base) {
                poll_once(&b, gen_id, &series, &tick);
                interval = Some(Interval::new(POLL_MS, move || {
                    sample_loss(gen_id);
                    poll_once(&b, gen_id, &series, &tick)
                }));
            }
            move || drop(interval)
        });
    }
    if !active {
        return html! {};
    }
    render_rows(&series.borrow())
}
