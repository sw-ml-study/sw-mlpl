//! `<LiveLossPanel>` -- the in-place loss chart shown beside the
//! "evaluating..." marker while a streamed connect-mode train runs.
//!
//! Mount starts a ~250ms poll of `loss_trace::seq` for THIS eval's
//! generation; each tick re-reads the streamed series and re-renders
//! the pure SVG, so points appear as `event: metric` frames land. The
//! poll interval IS the frame coalescing: however fast the server
//! emits, the DOM repaints at most 4x/second. Unmount (marker replaced
//! by the result) drops the interval; the durable record is the
//! `loss_trace::summary` line the result callback appends.

use gloo::timers::callback::Interval;
use yew::prelude::*;

use crate::loss_svg::loss_panel_svg;

const POLL_MS: u32 = 250;

#[function_component(LiveLossPanel)]
pub fn live_loss_panel() -> Html {
    // Read the generation EACH render (not captured at mount) for the
    // same reason TelemetryPanel does: a loaded demo reuses one panel
    // instance across lines, and a captured gen would stay stale.
    let gen_id = mlpl_web_eval::telemetry_trace::current_gen();
    let tick = use_state(|| 0u32);
    let active = mlpl_web_eval::telemetry_trace::is_remote(gen_id);
    {
        let tick = tick.clone();
        use_effect_with((active, gen_id), move |&(active, gen_id)| {
            let interval = active.then(|| {
                Interval::new(POLL_MS, move || {
                    tick.set(mlpl_web_eval::loss_trace::seq(gen_id));
                })
            });
            move || drop(interval)
        });
    }
    if !active {
        return html! {};
    }
    render_curve(gen_id)
}

/// The chart body: hidden until two streamed points exist (so the
/// panel never shows for metric-less evals), then the shared-axis SVG
/// with a step-count caption.
fn render_curve(gen_id: u32) -> Html {
    let (train, val) = mlpl_web_eval::loss_trace::series(gen_id);
    match loss_panel_svg(&train, &val) {
        None => html! {},
        Some(svg) => {
            let steps = train.len().max(val.len());
            html! {
                <div class="loss-panel">
                    { Html::from_html_unchecked(AttrValue::from(svg)) }
                    <div class="loss-caption">{ format!("live loss -- {steps} steps") }</div>
                </div>
            }
        }
    }
}

// ---- LiveLifePanel (Game of Life saga step 4) ----
//
// The live grid twin of LiveLossPanel: polls `frame_trace` for
// this eval's generation and repaints the board with the same
// `life` renderer the SMIL widget uses (T = 1 -> static grid),
// so live and persisted views cannot drift. Lives in this module
// (not a sibling file) to hold the crate at its module budget.
const LIFE_POLL_MS: u32 = 200;

#[function_component(LiveLifePanel)]
pub fn live_life_panel() -> Html {
    // Read the generation EACH render (see LiveLossPanel).
    let gen_id = mlpl_web_eval::telemetry_trace::current_gen();
    let tick = use_state(|| 0u32);
    let active = mlpl_web_eval::telemetry_trace::is_remote(gen_id);
    {
        let tick = tick.clone();
        use_effect_with((active, gen_id), move |&(active, gen_id)| {
            let interval = active.then(|| {
                Interval::new(LIFE_POLL_MS, move || {
                    tick.set(mlpl_web_eval::frame_trace::seq(gen_id));
                })
            });
            move || drop(interval)
        });
    }
    if !active {
        return html! {};
    }
    render_board(gen_id)
}

/// The grid body: hidden until a frame exists, then the rank-2
/// board rendered by the same `life` renderer the SMIL widget
/// uses (T = 1 -> static grid), with a generation caption.
fn render_board(gen_id: u32) -> Html {
    let Some((name, step, shape, values)) = mlpl_web_eval::frame_trace::latest(gen_id) else {
        return html! {};
    };
    let Ok(board) = mlpl_array::DenseArray::new(mlpl_array::Shape::new(shape), values) else {
        return html! {};
    };
    let Ok(svg) = mlpl_viz_marks::render_life(&board) else {
        return html! {};
    };
    html! {
        <div class="loss-panel life-panel">
            { Html::from_html_unchecked(AttrValue::from(svg)) }
            <div class="loss-caption">{ format!("live {name} -- generation {step}") }</div>
        </div>
    }
}
