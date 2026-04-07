use web_sys::{HtmlSelectElement, KeyboardEvent};
use yew::prelude::*;

use crate::demos::DEMOS;
use crate::state::DocTab;
use crate::tutorial::LESSONS;

const LANG_REFERENCE: &str = include_str!("../../../docs/lang-reference.md");
const USAGE_GUIDE: &str = include_str!("../../../docs/usage.md");

#[derive(Properties, PartialEq)]
pub struct UrlProps {
    pub url: &'static str,
}

#[function_component(GithubCorner)]
pub fn github_corner(props: &UrlProps) -> Html {
    html! {
        <a class="github-corner" href={props.url} aria-label="View source on GitHub" target="_blank" rel="noopener">
            <svg width="60" height="60" viewBox="0 0 250 250" aria-hidden="true">
                <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
                <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" class="octo-arm"></path>
                <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>
            </svg>
        </a>
    }
}

#[derive(Properties, PartialEq)]
pub struct HeaderProps {
    pub on_help: Callback<MouseEvent>,
    pub on_clear: Callback<MouseEvent>,
    pub on_demo: Callback<usize>,
    pub on_tutorial: Callback<MouseEvent>,
    pub tutorial_active: bool,
}

#[function_component(Header)]
pub fn header(props: &HeaderProps) -> Html {
    let on_demo = props.on_demo.clone();
    let on_change = Callback::from(move |e: Event| {
        let target: HtmlSelectElement = e.target_unchecked_into();
        let idx = target.value();
        if let Ok(i) = idx.parse::<usize>() {
            on_demo.emit(i);
            target.set_value("");
        }
    });
    let tutorial_label = if props.tutorial_active {
        "Exit Tutorial"
    } else {
        "Tutorial"
    };
    html! {
        <header>
            <h1>{"MLPL"}</h1>
            <span>{"v0.4 — Array Programming Language for ML"}</span>
            <div class="controls">
                <select class="demo-select" onchange={on_change} aria-label="Load demo">
                    <option value="" selected=true>{"Load Demo..."}</option>
                    { for DEMOS.iter().enumerate().map(|(i, d)| html!{
                        <option value={i.to_string()}>{ d.name }</option>
                    }) }
                </select>
                <button class="ctrl-btn" onclick={props.on_tutorial.clone()}>{ tutorial_label }</button>
                <button class="ctrl-btn" onclick={props.on_clear.clone()}>{"Clear"}</button>
                <button class="help-btn" onclick={props.on_help.clone()} aria-label="Show documentation" title="Documentation">{"?"}</button>
            </div>
        </header>
    }
}

#[derive(Properties, PartialEq)]
pub struct TutorialPanelProps {
    pub lesson_idx: usize,
    pub on_prev: Callback<MouseEvent>,
    pub on_next: Callback<MouseEvent>,
    pub on_run_example: Callback<String>,
    pub on_close: Callback<MouseEvent>,
}

#[function_component(TutorialPanel)]
pub fn tutorial_panel(props: &TutorialPanelProps) -> Html {
    let lesson = match LESSONS.get(props.lesson_idx) {
        Some(l) => l,
        None => return html! {},
    };
    let total = LESSONS.len();
    let is_first = props.lesson_idx == 0;
    let is_last = props.lesson_idx + 1 == total;

    let examples_html = lesson.examples.iter().map(|line| {
        let line_str = (*line).to_string();
        let on_click = {
            let on_run = props.on_run_example.clone();
            let line_str = line_str.clone();
            Callback::from(move |_| on_run.emit(line_str.clone()))
        };
        html! {
            <button class="lesson-example" onclick={on_click} title="Click to run">
                <span class="example-prompt">{"mlpl> "}</span>{ line }
            </button>
        }
    });

    html! {
        <div class="tutorial-panel">
            <div class="tutorial-header">
                <span class="tutorial-progress">{ format!("Lesson {} of {}", props.lesson_idx + 1, total) }</span>
                <h2>{ lesson.title }</h2>
                <button class="close-btn" onclick={props.on_close.clone()} aria-label="Exit tutorial">{"×"}</button>
            </div>
            <p class="tutorial-intro">{ lesson.intro }</p>
            <div class="lesson-examples">{ for examples_html }</div>
            <p class="tutorial-tryit"><strong>{"Try it: "}</strong>{ lesson.try_it }</p>
            <div class="tutorial-nav">
                <button class="ctrl-btn" disabled={is_first} onclick={props.on_prev.clone()}>{"← Previous"}</button>
                <button class="ctrl-btn" disabled={is_last} onclick={props.on_next.clone()}>{"Next →"}</button>
            </div>
        </div>
    }
}

#[derive(Properties, PartialEq)]
pub struct InputRowProps {
    pub value: String,
    pub on_input: Callback<InputEvent>,
    pub on_keydown: Callback<KeyboardEvent>,
}

#[function_component(InputRow)]
pub fn input_row(props: &InputRowProps) -> Html {
    html! {
        <div class="input-row">
            <span class="prompt">{"mlpl> "}</span>
            <input
                id="repl-input"
                type="text"
                autocomplete="off"
                spellcheck="false"
                value={props.value.clone()}
                oninput={props.on_input.clone()}
                onkeydown={props.on_keydown.clone()}
            />
        </div>
    }
}

#[function_component(Welcome)]
pub fn welcome() -> Html {
    html! {
        <div class="welcome">
            <p>{"Welcome to MLPL. Type expressions and press Enter."}</p>
            <p>{"Try: "}<code>{"1 + 2"}</code>{", "}<code>{"iota(5)"}</code>{", "}<code>{"reshape(iota(6), [2, 3])"}</code></p>
            <p>{"Type "}<code>{":help"}</code>{" for the function list, "}<code>{":clear"}</code>{" to reset, or click "}<code>{"?"}</code>{" for full docs."}</p>
        </div>
    }
}

#[function_component(Footer)]
pub fn footer(props: &UrlProps) -> Html {
    html! {
        <footer>
            <span>{"MLPL v0.4"}</span>
            <span class="footer-sep">{"·"}</span>
            <a href={props.url} target="_blank" rel="noopener">{"GitHub"}</a>
            <span class="footer-sep">{"·"}</span>
            <span>{"Built with Rust + Yew + WASM"}</span>
        </footer>
    }
}

#[derive(Properties, PartialEq)]
pub struct DocDialogProps {
    pub open: bool,
    pub on_close: Callback<MouseEvent>,
}

#[function_component(DocDialog)]
pub fn doc_dialog(props: &DocDialogProps) -> Html {
    let active_tab = use_state(|| DocTab::LangReference);

    if !props.open {
        return html! {};
    }

    let content = match *active_tab {
        DocTab::LangReference => LANG_REFERENCE,
        DocTab::Usage => USAGE_GUIDE,
    };
    let lang_class = if *active_tab == DocTab::LangReference {
        "tab active"
    } else {
        "tab"
    };
    let usage_class = if *active_tab == DocTab::Usage {
        "tab active"
    } else {
        "tab"
    };

    let select_lang = {
        let active_tab = active_tab.clone();
        Callback::from(move |_| active_tab.set(DocTab::LangReference))
    };
    let select_usage = {
        let active_tab = active_tab.clone();
        Callback::from(move |_| active_tab.set(DocTab::Usage))
    };
    let stop_propagation = Callback::from(|e: MouseEvent| e.stop_propagation());

    html! {
        <div class="modal-backdrop" onclick={props.on_close.clone()}>
            <div class="modal" onclick={stop_propagation}>
                <div class="modal-header">
                    <div class="tabs">
                        <button class={lang_class} onclick={select_lang}>{"Language Reference"}</button>
                        <button class={usage_class} onclick={select_usage}>{"Usage Guide"}</button>
                    </div>
                    <button class="close-btn" onclick={props.on_close.clone()} aria-label="Close">{"×"}</button>
                </div>
                <div class="modal-body">
                    <pre class="doc-content">{ content }</pre>
                </div>
            </div>
        </div>
    }
}
