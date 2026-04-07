mod components;
mod handlers;
mod help;
mod state;

use components::{DocDialog, Footer, GithubCorner, Header, InputRow, Welcome};
use handlers::{EvalDeps, make_keydown, make_oninput, make_submit};
use mlpl_wasm::WasmSession;
use state::HistoryEntry;
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

    let on_submit = make_submit(EvalDeps {
        session,
        history: history.clone(),
        input_value: input_value.clone(),
        cmd_history: cmd_history.clone(),
        cmd_index: cmd_index.clone(),
    });

    use_effect_with(history.clone(), |_| {
        scroll_and_focus();
        || ()
    });

    render(
        on_submit,
        input_value,
        cmd_history,
        cmd_index,
        history,
        dialog_open,
    )
}

fn render(
    on_submit: Callback<String>,
    input_value: UseStateHandle<String>,
    cmd_history: UseStateHandle<Vec<String>>,
    cmd_index: UseStateHandle<Option<usize>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
    dialog_open: UseStateHandle<bool>,
) -> Html {
    let on_input = make_oninput(input_value.clone());
    let on_keydown = make_keydown(on_submit, input_value.clone(), cmd_history, cmd_index);
    let open_dialog = toggle(dialog_open.clone(), true);
    let close_dialog = toggle(dialog_open.clone(), false);

    html! {
        <>
            <GithubCorner url={REPO_URL} />
            <Header on_help={open_dialog} />
            <main>
                <div id="output" class="output">
                    <Welcome />
                    { for history.iter().map(render_entry) }
                </div>
                <InputRow value={(*input_value).clone()} on_input={on_input} on_keydown={on_keydown} />
            </main>
            <Footer url={REPO_URL} />
            <DocDialog open={*dialog_open} on_close={close_dialog} />
        </>
    }
}

fn toggle(handle: UseStateHandle<bool>, value: bool) -> Callback<MouseEvent> {
    Callback::from(move |_| handle.set(value))
}

fn render_entry(entry: &HistoryEntry) -> Html {
    let output_class = if entry.is_error {
        "output-line error"
    } else {
        "output-line"
    };
    html! {
        <div class="entry">
            <div class="input-line"><span class="prompt">{"mlpl> "}</span>{ &entry.input }</div>
            <pre class={output_class}>{ &entry.output }</pre>
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

fn main() {
    yew::Renderer::<App>::new().render();
}
