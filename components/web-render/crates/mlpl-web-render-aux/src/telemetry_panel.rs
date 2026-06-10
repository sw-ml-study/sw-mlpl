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

/// Ring buffers of the four metric percentages + the freshest snapshot
/// + the last fetch error (shown when polling fails) + a monotonic seq
/// the component mirrors into a `use_state` to force a re-render.
#[derive(Default)]
struct Series {
    rows: [Vec<u32>; 4],
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
fn poll_once(base: &str, series: &Rc<RefCell<Series>>, tick: &UseStateHandle<u32>) {
    let url = format!("{}/v1/stats", base.trim_end_matches('/'));
    let series = series.clone();
    let tick = tick.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let outcome = fetch_snapshot(&url).await;
        let seq = {
            let mut s = series.borrow_mut();
            match outcome {
                Ok(snap) => {
                    // Mirror into the shared trace so the sparkline
                    // PERSISTS in the result after the marker unmounts.
                    mlpl_web_eval::telemetry_trace::push(&snap);
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
        .map(|(i, buf)| {
            html! {
                <span class="telemetry-metric">
                    <span class="telemetry-label">{ LABELS[i] }</span>
                    <span class="telemetry-spark">{ sparkline(buf, 100) }</span>
                    <span class="telemetry-val">{ format!("{}%", pcts[i]) }</span>
                </span>
            }
        })
        .collect::<Html>();
    let note = series.note.as_ref().map_or_else(
        || html! {},
        |n| html! { <span class="telemetry-note">{ format!("(stats {n})") }</span> },
    );
    html! {
        <div class="telemetry-panel">
            <div class="telemetry-line">{ metrics }{ note }</div>
        </div>
    }
}

#[function_component(TelemetryPanel)]
pub fn telemetry_panel() -> Html {
    // Only show for evals that actually run on the SERVER -- a browser
    // -local eval (CPU-tier demo) would otherwise display server CPU
    // that has nothing to do with where it runs. Both conditions are
    // stable for the panel's lifetime, so the early return is safe
    // before the hooks.
    let connected = mlpl_web_eval::eval::current_connect_url_from_window();
    let (Some(base), true) = (connected, mlpl_web_eval::telemetry_trace::is_remote()) else {
        return html! {};
    };
    let series = use_mut_ref(Series::default);
    let tick = use_state(|| 0u32);
    {
        let series = series.clone();
        let tick = tick.clone();
        use_effect_with((), move |_| {
            // The trace was reset by the dispatcher (begin) before this
            // marker; just start polling.
            poll_once(&base, &series, &tick);
            let interval = Interval::new(POLL_MS, move || poll_once(&base, &series, &tick));
            move || drop(interval)
        });
    }
    render_rows(&series.borrow())
}
