use wasm_bindgen::JsCast;
use yew::prelude::*;

struct TourStep {
    target: &'static str,
    title: &'static str,
    body: &'static str,
    click_before: &'static str,
}

const STEPS: &[TourStep] = &[
    TourStep {
        target: "repl-input",
        title: "REPL Input",
        body: "Type expressions here and press Enter to evaluate. Try 1 + 2 or iota(5).",
        click_before: "",
    },
    TourStep {
        target: "demo-select",
        title: "Demos",
        body: "Pick a pre-built demo from this dropdown. Each runs a complete example with intro and takeaway narration.",
        click_before: "",
    },
    TourStep {
        target: "help-btn",
        title: "Documentation",
        body: "Click ? to open the docs panel. Let's look inside.",
        click_before: "",
    },
    TourStep {
        target: "doc-tab-reference",
        title: "Language Reference",
        body: "Full syntax, every builtin function, and operator reference in one scrollable page.",
        click_before: "[data-tour-target='help-btn']",
    },
    TourStep {
        target: "doc-tab-usage",
        title: "Usage Guide",
        body: "How to use the playground: REPL commands, keyboard shortcuts, demo runner, and connect mode.",
        click_before: "[data-tour-target='doc-tab-usage']",
    },
    TourStep {
        target: "doc-tab-glossary",
        title: "Glossary",
        body: "Searchable glossary of ML and array-programming terms. Click any [[linked term]] in lessons or demos to jump here.",
        click_before: "[data-tour-target='doc-tab-glossary']",
    },
    TourStep {
        target: "doc-tab-diagrams",
        title: "Diagrams",
        body: "Architecture and concept diagrams. Visual companions to the lessons.",
        click_before: "[data-tour-target='doc-tab-diagrams']",
    },
    TourStep {
        target: "tab-tutorial",
        title: "Tutorial",
        body: "Guided lessons from arithmetic to transformers. Let's open it.",
        click_before: ".modal-backdrop",
    },
    TourStep {
        target: "tutorial-panel",
        title: "Lesson Index",
        body: "Pick any lesson to start. Each has explanations and runnable code examples in the REPL pane on the right.",
        click_before: "[data-tour-target='tab-tutorial']",
    },
    TourStep {
        target: "tab-paths",
        title: "Learning Paths",
        body: "Structured sequences of lessons, demos, glossary entries, and diagrams. Let's open one.",
        click_before: "",
    },
    TourStep {
        target: "path-card-first",
        title: "Pick a Path",
        body: "Each card is a curated walk-through. Let's open the Dimensionality Reduction path.",
        click_before: "[data-tour-target='tab-paths']",
    },
    TourStep {
        target: "paths-walker-body",
        title: "Walking a Path",
        body: "You're now inside the path. Each step shows a lesson, demo, glossary entry, or diagram with context. Use the arrow buttons to navigate.",
        click_before: "[data-tour-target='path-card-first']",
    },
    TourStep {
        target: "path-open-lesson",
        title: "Open a Lesson from a Path",
        body: "Click this button to open the lesson in the Tutorial pane. The path remembers where you are.",
        click_before: "",
    },
    TourStep {
        target: "back-to-path",
        title: "Return to Your Path",
        body: "After reading the lesson, click this button to jump back to the path at the step you left off. This is how you navigate between paths and lessons without losing your place.",
        click_before: "[data-tour-target='path-open-lesson']",
    },
    TourStep {
        target: "paths-walker-body",
        title: "Back on the Path",
        body: "You're back on the path right where you left off. The path always remembers your position.",
        click_before: "[data-tour-target='back-to-path']",
    },
    TourStep {
        target: "repl-input",
        title: "Autocomplete",
        body: "Press Ctrl+Space for autocomplete suggestions. Arrow keys to navigate, Enter to accept.",
        click_before: "[data-tour-target='tab-repl']",
    },
];

pub const STEP_COUNT: usize = STEPS.len();

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
    let click_sel = s.click_before;
    let step = props.step;
    let settled = use_state(|| false);
    {
        let settled = settled.clone();
        use_effect_with(step, move |_| {
            settled.set(false);
            fire_click(click_sel);
            let s2 = settled.clone();
            gloo::timers::callback::Timeout::new(50, move || s2.set(true)).forget();
            || ()
        });
    }
    let pos = target_position(s.target);
    let style = tooltip_style(&pos);
    let counter = format!("{} / {}", props.step + 1, STEPS.len());
    let done = props.step + 1 >= STEPS.len();
    html! {
        <>
            <div class="tour-backdrop" />
            <div class="tour-spotlight" style={spotlight_style(&pos)} />
            <div class="tour-tooltip" style={style}>
                <div class="tour-header">
                    <strong>{ s.title }</strong>
                    <button class="tour-close" onclick={props.on_close.clone()} aria-label="Close tour">{"\u{00d7}"}</button>
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

fn fire_click(selector: &str) {
    if selector.is_empty() {
        return;
    }
    if let Some(el) = query(selector) {
        el.unchecked_into::<web_sys::HtmlElement>().click();
    }
}

struct Rect {
    top: f64,
    left: f64,
    width: f64,
    height: f64,
}

fn target_position(target: &str) -> Rect {
    query(&format!("[data-tour-target='{target}']"))
        .map(|el| {
            let r = el.get_bounding_client_rect();
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

fn query(sel: &str) -> Option<web_sys::Element> {
    web_sys::window()?.document()?.query_selector(sel).ok()?
}

fn spotlight_style(r: &Rect) -> String {
    let p = 6.0;
    format!(
        "top:{}px;left:{}px;width:{}px;height:{}px",
        r.top - p,
        r.left - p,
        r.width + p * 2.0,
        r.height + p * 2.0
    )
}

fn tooltip_style(r: &Rect) -> String {
    let (vw, vh) = viewport_size();
    let (tw, th) = (340.0, 180.0);
    let below = r.top + r.height + 12.0;
    let top = if below + th > vh {
        (r.top - th - 12.0).max(8.0)
    } else {
        below
    };
    let left = r.left.max(8.0).min(vw - tw - 8.0);
    format!("top:{top}px;left:{left}px")
}

fn viewport_size() -> (f64, f64) {
    web_sys::window()
        .map(|w| {
            let vw = w
                .inner_width()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(1200.0);
            let vh = w
                .inner_height()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(800.0);
            (vw, vh)
        })
        .unwrap_or((1200.0, 800.0))
}
