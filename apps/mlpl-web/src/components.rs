use web_sys::{HtmlSelectElement, KeyboardEvent};
use yew::prelude::*;

use crate::demos::DEMOS;
use crate::glossary_view::GlossaryView;
pub use crate::tutorial::{TutorialPanel, TutorialPanelProps, TutorialView};
use mlpl_web_eval::state::DocTab;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeaderMode {
    Repl,
    Tutorial,
    Paths,
}

#[derive(Properties, PartialEq)]
pub struct HeaderProps {
    pub on_help: Callback<MouseEvent>,
    pub on_select_repl: Callback<MouseEvent>,
    pub on_select_tutorial: Callback<MouseEvent>,
    pub on_select_paths: Callback<MouseEvent>,
    pub mode: HeaderMode,
}

#[function_component(Header)]
pub fn header(props: &HeaderProps) -> Html {
    let cls = |m: HeaderMode| {
        if props.mode == m { "tab active" } else { "tab" }
    };
    html! {
        <header>
            <h1><img src="mlpl-badge.webp" alt="" class="title-badge" />{"sw-MLPL"}</h1>
            <div class="title-text">
                <span class="title-line">{"Software Wrighter's Machine Learning Programming Language Playground"}</span>
                <span class="title-subtitle">{"Array Programming for Machine Learning"}</span>
            </div>
            <div class="header-tabs">
                <button class={cls(HeaderMode::Repl)} onclick={props.on_select_repl.clone()}>{"REPL"}</button>
                <button class={cls(HeaderMode::Tutorial)} onclick={props.on_select_tutorial.clone()}>{"Tutorial"}</button>
                <button class={cls(HeaderMode::Paths)} onclick={props.on_select_paths.clone()}>{"Paths"}</button>
            </div>
            <button class="help-btn" onclick={props.on_help.clone()} aria-label="Show documentation" title="Documentation">{"?"}</button>
        </header>
    }
}

#[derive(Properties, PartialEq)]
pub struct ModeBarProps {
    pub on_clear: Callback<MouseEvent>,
    pub on_demo: Callback<usize>,
    pub on_upload: Callback<web_sys::Event>,
    /// Saga 29 step 016: cancel handler for the `<input
    /// type=file>`'s `cancel` event. Binds `Err("cancelled")`
    /// under the pending upload name when the user dismisses
    /// the file picker.
    pub on_upload_cancel: Callback<web_sys::Event>,
    /// Saga 29 step 016: lifted to the parent so the
    /// `:upload <name>` REPL command handler in handlers.rs
    /// can also click() the input programmatically.
    pub upload_input_ref: NodeRef,
    pub tutorial_active: bool,
}

#[function_component(ModeBar)]
pub fn mode_bar(props: &ModeBarProps) -> Html {
    let on_demo = props.on_demo.clone();
    let on_change = Callback::from(move |e: Event| {
        let target: HtmlSelectElement = e.target_unchecked_into();
        let idx = target.value();
        if let Ok(i) = idx.parse::<usize>() {
            on_demo.emit(i);
            target.set_value("");
        }
    });
    let cls = if props.tutorial_active {
        "modebar tutorial"
    } else {
        "modebar repl"
    };
    let demo_dropdown = if props.tutorial_active {
        html! {}
    } else {
        // Saga 29 step 010 follow-up: render the dropdown
        // alphabetically by demo name (case-insensitive).
        let mut sorted: Vec<(usize, &str)> =
            DEMOS.iter().enumerate().map(|(i, d)| (i, d.name)).collect();
        sorted.sort_by_key(|(_, name)| name.to_ascii_lowercase());
        html! {
            <select class="demo-select" onchange={on_change} aria-label="Load demo">
                <option value="" selected=true>{"Load Demo..."}</option>
                { for sorted.iter().map(|(i, name)| html!{
                    <option value={i.to_string()}>{ *name }</option>
                }) }
            </select>
        }
    };
    let clear_label = if props.tutorial_active {
        "Reset Tutorial"
    } else {
        "Reset REPL"
    };
    // Saga 29 step 011 follow-up: hidden file input + visible
    // "Upload Image" button. Step 016: input ref lifted to the
    // parent so :upload <name> REPL command can click() it,
    // and the cancel handler binds Err("cancelled") on
    // dismiss.
    let upload_widget = if props.tutorial_active {
        html! {}
    } else {
        let input_ref = props.upload_input_ref.clone();
        let on_click = Callback::from(move |_: MouseEvent| {
            if let Some(input) = input_ref.cast::<web_sys::HtmlInputElement>() {
                input.click();
            }
        });
        html! {
            <>
                <input
                    ref={props.upload_input_ref.clone()}
                    type="file"
                    accept="image/*"
                    style="display: none"
                    onchange={props.on_upload.clone()}
                    oncancel={props.on_upload_cancel.clone()}
                />
                <button class="ctrl-btn" onclick={on_click} title="Upload a photo (resized to 64x64). Binds `uploaded = Ok({pixels, h, w})` on success, `Err(\"cancelled\")` on dismiss.">
                    {"Upload Image"}
                </button>
            </>
        }
    };
    html! {
        <div class={cls}>
            { demo_dropdown }
            { upload_widget }
            <button class="ctrl-btn" onclick={props.on_clear.clone()}>{ clear_label }</button>
        </div>
    }
}

#[derive(Properties, PartialEq)]
pub struct InputRowProps {
    pub value: String,
    pub on_input: Callback<InputEvent>,
    pub on_keydown: Callback<KeyboardEvent>,
    pub in_tutorial: bool,
}

#[function_component(InputRow)]
pub fn input_row(props: &InputRowProps) -> Html {
    let (label_text, label_class) = if props.in_tutorial {
        ("(Tutorial)", "session-label tutorial")
    } else {
        ("(REPL)", "session-label repl")
    };
    html! {
        <div class="input-wrap">
            <div class={label_class}>{ label_text }</div>
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
    // Same shape as the Software Wrighter web-sw-cor24-* live demos:
    // legal blurb, repo, project links, and build provenance --
    // separated by middots. Build env vars come from `build.rs`
    // and refresh whenever .git/HEAD changes.
    let build_info = format!(
        "{} \u{00b7} {} \u{00b7} {}",
        env!("BUILD_HOST"),
        env!("BUILD_SHA"),
        env!("BUILD_TIMESTAMP"),
    );
    html! {
        <footer>
            <span>{"MIT License"}</span>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <span>{"\u{00a9} 2026 Michael A Wright"}</span>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href={props.url} target="_blank" rel="noopener">{"GitHub"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://github.com/sw-ml-study/sw-mlpl/blob/main/CHANGES.md" target="_blank" rel="noopener">{"Changes"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://software-wrighter-lab.github.io/" target="_blank" rel="noopener">{"Blog"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://discord.com/invite/Ctzk5uHggZ" target="_blank" rel="noopener">{"Discord"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://www.youtube.com/@SoftwareWrighter" target="_blank" rel="noopener">{"YouTube"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <span>{ build_info }</span>
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

    let body = match *active_tab {
        DocTab::LangReference => html! { <pre class="doc-content">{ LANG_REFERENCE }</pre> },
        DocTab::Usage => html! { <pre class="doc-content">{ USAGE_GUIDE }</pre> },
        DocTab::Glossary => html! { <GlossaryView /> },
        DocTab::Diagrams => html! { <crate::diagrams_view::DiagramsView /> },
    };
    let cls = |t: DocTab| {
        if *active_tab == t {
            "tab active"
        } else {
            "tab"
        }
    };
    let setter = |t: DocTab| {
        let h = active_tab.clone();
        Callback::from(move |_| h.set(t))
    };
    let stop = Callback::from(|e: MouseEvent| e.stop_propagation());
    html! {
        <div class="modal-backdrop" onclick={props.on_close.clone()}>
            <div class="modal" onclick={stop}>
                <div class="modal-header">
                    <div class="tabs">
                        <button class={cls(DocTab::LangReference)} onclick={setter(DocTab::LangReference)}>{"Language Reference"}</button>
                        <button class={cls(DocTab::Usage)} onclick={setter(DocTab::Usage)}>{"Usage Guide"}</button>
                        <button class={cls(DocTab::Glossary)} onclick={setter(DocTab::Glossary)}>{"Glossary"}</button>
                        <button class={cls(DocTab::Diagrams)} onclick={setter(DocTab::Diagrams)}>{"Diagrams"}</button>
                    </div>
                    <button class="close-btn" onclick={props.on_close.clone()} aria-label="Close">{"×"}</button>
                </div>
                <div class="modal-body">{ body }</div>
            </div>
        </div>
    }
}
