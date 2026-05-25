use wasm_bindgen::JsCast;
use yew::prelude::*;

struct TourStep {
    target: &'static str,
    title: &'static str,
    body: &'static str,
}

const STEPS: &[TourStep] = &[
    TourStep {
        target: "repl-input",
        title: "REPL Input",
        body: "Type expressions here. Press Enter to evaluate. Try 1 + 2 or iota(5).",
    },
    TourStep {
        target: "demo-select",
        title: "Demos",
        body: "Load a pre-built demo. Each runs a complete example with narration.",
    },
    TourStep {
        target: "tab-tutorial",
        title: "Tutorial",
        body: "Follow guided lessons, from arithmetic to transformers.",
    },
    TourStep {
        target: "tab-paths",
        title: "Learning Paths",
        body: "Structured sequences of lessons, demos, and glossary entries.",
    },
    TourStep {
        target: "help-btn",
        title: "Documentation",
        body: "Full language reference, usage guide, glossary, and architecture diagrams.",
    },
    TourStep {
        target: "repl-input",
        title: "Autocomplete",
        body: "Press Ctrl+Space for autocomplete suggestions. Arrow keys to navigate, Enter to accept.",
    },
];

#[derive(Properties, PartialEq)]
pub struct TourProps {
    pub step: usize,
    pub on_next: Callback<MouseEvent>,
    pub on_prev: Callback<MouseEvent>,
    pub on_close: Callback<MouseEvent>,
}

#[function_component(TourTooltip)]
pub fn tour_tooltip(props: &TourProps) -> Html {
    let s = &STEPS[props.step.min(STEPS.len() - 1)];
    let pos = target_position(s.target);
    let style = tooltip_style(&pos, s.target);
    let counter = format!("{} / {}", props.step + 1, STEPS.len());
    let done = props.step + 1 >= STEPS.len();
    html! {
        <>
            <div class="tour-backdrop" />
            <div class="tour-spotlight" style={spotlight_style(&pos)} />
            <div class="tour-tooltip" style={style}>
                <div class="tour-header">
                    <strong>{ s.title }</strong>
                    <button class="tour-close" onclick={props.on_close.clone()} aria-label="Close tour">{"x"}</button>
                </div>
                <p class="tour-body">{ s.body }</p>
                <div class="tour-nav">
                    <span class="tour-counter">{ counter }</span>
                    <button class="tour-btn" onclick={props.on_prev.clone()} disabled={props.step == 0}>{"Back"}</button>
                    <button class="tour-btn tour-btn-primary" onclick={props.on_next.clone()}>
                        { if done { "Done" } else { "Next" } }
                    </button>
                </div>
            </div>
        </>
    }
}

struct Rect {
    top: f64,
    left: f64,
    width: f64,
    height: f64,
}

fn target_position(target: &str) -> Rect {
    let sel = format!("[data-tour-target='{target}']");
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.query_selector(&sel).ok()?)
        .map(|el| {
            let r = el
                .unchecked_into::<web_sys::Element>()
                .get_bounding_client_rect();
            Rect {
                top: r.top(),
                left: r.left(),
                width: r.width(),
                height: r.height(),
            }
        })
        .unwrap_or(Rect {
            top: 100.0,
            left: 100.0,
            width: 200.0,
            height: 40.0,
        })
}

fn spotlight_style(r: &Rect) -> String {
    let pad = 6.0;
    format!(
        "top:{}px;left:{}px;width:{}px;height:{}px",
        r.top - pad,
        r.left - pad,
        r.width + pad * 2.0,
        r.height + pad * 2.0,
    )
}

fn tooltip_style(r: &Rect, _target: &str) -> String {
    let vh = web_sys::window()
        .and_then(|w| w.inner_height().ok())
        .and_then(|v| v.as_f64())
        .unwrap_or(800.0);
    let below = r.top + r.height + 12.0;
    let top = if below + 180.0 > vh {
        (r.top - 180.0 - 12.0).max(8.0)
    } else {
        below
    };
    let left = r.left.max(8.0);
    format!("top:{top}px;left:{left}px")
}
