// On a native (non-wasm) target the entire Yew/wasm-bindgen frontend
// is dead code -- only the stub `main` below runs. Suppress the
// resulting dead_code / unused-import warnings so `cargo build` and
// `cargo clippy -D warnings` stay clean for both targets.
#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, unused_imports, unused_variables)
)]

mod components;
mod demos;
mod handlers;
mod help;
mod state;
mod summary;
mod tutorial;

use components::{
    DocDialog, Footer, GithubCorner, Header, InputRow, TutorialPanel, TutorialPanelProps, Welcome,
};
use handlers::{
    EvalDeps, make_clear, make_keydown, make_oninput, make_run_demo, make_submit, toggle_bool,
};
use mlpl_wasm::WasmSession;
use state::{EntryKind, HistoryEntry};
use tutorial::{run_example, step_lesson, toggle_tutorial};
use wasm_bindgen::JsCast;
use web_sys::HtmlInputElement;
use yew::prelude::*;

const REPO_URL: &str = "https://github.com/sw-ml-study/sw-mlpl";

#[function_component(App)]
fn app() -> Html {
    let session = use_mut_ref(WasmSession::new);
    let history = use_state(Vec::<HistoryEntry>::new);
    let input_value = use_state(String::new);
    let cmd_history = use_state(Vec::<String>::new);
    let cmd_index = use_state(|| None::<usize>);
    let dialog_open = use_state(|| false);
    let lesson_idx = use_state(|| None::<usize>);

    let on_submit = make_submit(EvalDeps {
        session: session.clone(),
        history: history.clone(),
        input_value: input_value.clone(),
        cmd_history: cmd_history.clone(),
        cmd_index: cmd_index.clone(),
    });
    let on_clear = make_clear(session.clone(), history.clone());
    let on_demo = make_run_demo(session, history.clone());

    use_effect_with(history.clone(), |_| {
        scroll_and_focus();
        || ()
    });

    render(RenderArgs {
        on_submit,
        on_clear,
        on_demo,
        input_value,
        cmd_history,
        cmd_index,
        history,
        dialog_open,
        lesson_idx,
    })
}

struct RenderArgs {
    on_submit: Callback<String>,
    on_clear: Callback<MouseEvent>,
    on_demo: Callback<usize>,
    input_value: UseStateHandle<String>,
    cmd_history: UseStateHandle<Vec<String>>,
    cmd_index: UseStateHandle<Option<usize>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
    dialog_open: UseStateHandle<bool>,
    lesson_idx: UseStateHandle<Option<usize>>,
}

fn render(a: RenderArgs) -> Html {
    let on_input = make_oninput(a.input_value.clone());
    let on_keydown = make_keydown(
        a.on_submit.clone(),
        a.input_value.clone(),
        a.cmd_history,
        a.cmd_index,
    );
    let open_dialog = toggle_bool(a.dialog_open.clone(), true);
    let close_dialog = toggle_bool(a.dialog_open.clone(), false);
    let cur_lesson = *a.lesson_idx;
    let tutorial_active = cur_lesson.is_some();
    let on_tutorial = toggle_tutorial(a.lesson_idx.clone());
    let on_run_example = run_example(a.on_submit.clone(), a.input_value.clone());

    html! {
        <>
            <GithubCorner url={REPO_URL} />
            <Header on_help={open_dialog} on_clear={a.on_clear} on_demo={a.on_demo} on_tutorial={on_tutorial} {tutorial_active} />
            <main>
                { render_tutorial(cur_lesson, a.lesson_idx.clone(), on_run_example) }
                <div id="output" class="output">
                    { if tutorial_active { html!{} } else { html!{ <Welcome /> } } }
                    { for a.history.iter().map(render_entry) }
                </div>
                <InputRow value={(*a.input_value).clone()} on_input={on_input} on_keydown={on_keydown} />
            </main>
            <Footer url={REPO_URL} />
            <DocDialog open={*a.dialog_open} on_close={close_dialog} />
        </>
    }
}

fn render_tutorial(
    cur: Option<usize>,
    lesson: UseStateHandle<Option<usize>>,
    on_run_example: Callback<String>,
) -> Html {
    let Some(idx) = cur else {
        return html! {};
    };
    let on_prev = step_lesson(lesson.clone(), -1);
    let on_next = step_lesson(lesson.clone(), 1);
    let on_close = Callback::from(move |_| lesson.set(None));
    let props = TutorialPanelProps {
        lesson_idx: idx,
        on_prev,
        on_next,
        on_run_example,
        on_close,
    };
    html! { <TutorialPanel ..props /> }
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let safe = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if safe {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn render_entry(entry: &HistoryEntry) -> Html {
    if entry.kind == EntryKind::Narration {
        // Demo narration: prose framing around the code output.
        // No `mlpl>` prompt, no output pre-formatting; `input` is
        // the heading (e.g. "About this demo" / "What just
        // happened"), `output` is the narration body.
        return html! {
            <div class="narration">
                <div class="narration-heading">{ &entry.input }</div>
                <div class="narration-body">{ &entry.output }</div>
            </div>
        };
    }
    let body = if !entry.is_error && entry.output.trim_start().starts_with("<svg") {
        let svg_html = Html::from_html_unchecked(AttrValue::from(entry.output.clone()));
        let href = format!(
            "data:image/svg+xml;charset=utf-8,{}",
            percent_encode(&entry.output)
        );
        html! {
            <div class="svg-output">
                { svg_html }
                <a class="svg-download" href={href} download="mlpl.svg" title="Download SVG" aria-label="Download SVG">{"⬇"}</a>
            </div>
        }
    } else if entry.is_error {
        html! { <pre class={"output-line error"}>{ &entry.output }</pre> }
    } else if let Some(s) = summary::summarize(&entry.output) {
        let summary_text = format!(
            "{}  min={}  max={}  mean={}  median={}  std={}",
            s.shape,
            summary::fmt_stat(s.min),
            summary::fmt_stat(s.max),
            summary::fmt_stat(s.mean),
            summary::fmt_stat(s.median),
            summary::fmt_stat(s.std),
        );
        html! {
            <details class="output-summary">
                <summary>{ summary_text }</summary>
                <pre class={"output-line"}>{ &entry.output }</pre>
            </details>
        }
    } else {
        html! { <pre class={"output-line"}>{ &entry.output }</pre> }
    };
    html! {
        <div class="entry">
            <div class="input-line"><span class="prompt">{"mlpl> "}</span>{ &entry.input }</div>
            { body }
        </div>
    }
}

fn scroll_and_focus() {
    if let Some(window) = web_sys::window()
        && let Some(document) = window.document()
    {
        if let Some(el) = document.get_element_by_id("output") {
            el.set_scroll_top(el.scroll_height());
        }
        if let Some(el) = document.get_element_by_id("repl-input")
            && let Ok(input) = el.dyn_into::<HtmlInputElement>()
        {
            let _ = input.focus();
        }
    }
}

// mlpl-web is a Yew/WASM bundle served via `trunk serve` (or built
// into ./pages/ via scripts/build-pages.sh). It is *not* a native
// CLI -- linking succeeds for the host target but the resulting
// binary panics on first call into js-sys because the imported
// statics only resolve inside a wasm runtime. Provide a friendly
// stub when invoked as a native binary so users get a clear error
// instead of a stack trace, and skip mlpl-web from sw-install via
// `sw-install -p . --bin mlpl-repl --bin mlpl-lab`.

#[cfg(target_arch = "wasm32")]
fn main() {
    yew::Renderer::<App>::new().render();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "mlpl-web is a WASM/Yew bundle, not a native CLI.\n\
         \n\
         To run it locally:\n\
           cd apps/mlpl-web && trunk serve\n\
         \n\
         To rebuild the deployed pages/ bundle:\n\
           ./scripts/build-pages.sh\n\
         \n\
         Or visit the live demo: https://sw-ml-study.github.io/sw-mlpl/"
    );
    std::process::exit(1);
}
