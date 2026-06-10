//! `<TelemetryPanel>` -- live backend CPU/GPU/RAM/VRAM sparklines shown
//! beneath the "evaluating..." marker during a connect-mode eval.
//!
//! Mount starts a ~2.5s poll of `GET /v1/stats`; unmount (when the
//! marker is replaced by the result) drops the interval, so polling is
//! bounded to the lifetime of the running line. Renders nothing in
//! local mode (no connected backend to poll). The sample buffers live
//! in a `RefCell` (the poll closure appends without the stale
//! state-handle problem); a bumped `use_state` seq drives the re-render.

use std::cell::RefCell;
use std::rc::Rc;

use gloo::timers::callback::Interval;
use mlpl_monitor_types::Snapshot;
use mlpl_monitor_types::spark::{metric_percents, sparkline};
use yew::prelude::*;

const POLL_MS: u32 = 2500;
const WINDOW: usize = 24;
const LABELS: [&str; 4] = ["CPU ", "RAM ", "GPU ", "VRAM"];

/// Ring buffers of the four metric percentages + the freshest snapshot
/// (for the numeric labels) + a monotonic seq the component mirrors
/// into a `use_state` to force a re-render on each new sample.
#[derive(Default)]
struct Series {
    rows: [Vec<u32>; 4],
    latest: Option<Snapshot>,
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
        self.seq = self.seq.wrapping_add(1);
    }
}

/// One-shot `GET <base>/v1/stats`; on success appends to the buffers
/// and bumps `tick` (its new value forces the component to re-render).
fn poll_once(base: &str, series: &Rc<RefCell<Series>>, tick: &UseStateHandle<u32>) {
    let url = format!("{}/v1/stats", base.trim_end_matches('/'));
    let series = series.clone();
    let tick = tick.clone();
    wasm_bindgen_futures::spawn_local(async move {
        if let Ok(resp) = gloo::net::http::Request::get(&url).send().await {
            if let Ok(snap) = resp.json::<Snapshot>().await {
                series.borrow_mut().push(snap);
                let seq = series.borrow().seq;
                tick.set(seq);
            }
        }
    });
}

fn render_rows(series: &Series) -> Html {
    let pcts = series.latest.as_ref().map_or([0u32; 4], metric_percents);
    let rows = series
        .rows
        .iter()
        .enumerate()
        .map(|(i, buf)| {
            html! {
                <div class="telemetry-row">
                    <span class="telemetry-label">{ LABELS[i] }</span>
                    <span class="telemetry-spark">{ sparkline(buf, 100) }</span>
                    <span class="telemetry-val">{ format!("{}%", pcts[i]) }</span>
                </div>
            }
        })
        .collect::<Html>();
    html! { <div class="telemetry-panel">{ rows }</div> }
}

#[function_component(TelemetryPanel)]
pub fn telemetry_panel() -> Html {
    let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
        return html! {};
    };
    let series = use_mut_ref(Series::default);
    let tick = use_state(|| 0u32);
    {
        let series = series.clone();
        let tick = tick.clone();
        use_effect_with((), move |_| {
            poll_once(&base, &series, &tick);
            let interval = Interval::new(POLL_MS, move || poll_once(&base, &series, &tick));
            move || drop(interval)
        });
    }
    render_rows(&series.borrow())
}
