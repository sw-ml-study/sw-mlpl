use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{FileReader, HtmlInputElement, HtmlTextAreaElement};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct EditorProps {
    pub content: String,
    pub on_change: Callback<String>,
    pub on_run: Callback<MouseEvent>,
    pub on_save: Callback<MouseEvent>,
    pub on_clear: Callback<MouseEvent>,
}

#[function_component(EditorPanel)]
pub fn editor_panel(props: &EditorProps) -> Html {
    let file_ref = use_node_ref();
    let on_input = make_input_handler(props.on_change.clone());
    let on_keydown = make_keydown_handler(props.on_run.clone());
    html! {
        <div class="editor-panel">
            { toolbar(props, &file_ref) }
            <textarea
                class="editor-textarea"
                spellcheck="false"
                placeholder={"# Type or paste MLPL script here\nx = range(10)\nreshape(x, [2, 5])"}
                value={props.content.clone()}
                oninput={on_input}
                onkeydown={on_keydown}
            />
        </div>
    }
}

/// The Run / Load / Save / Clear button row (plus the hidden file
/// input the Load button forwards to).
fn toolbar(props: &EditorProps, file_ref: &NodeRef) -> Html {
    let on_load_click = make_load_click_handler(file_ref.clone());
    let on_file_change = make_file_change_handler(props.on_change.clone());
    html! {
        <div class="editor-toolbar">
            <button class="editor-btn editor-btn-run" onclick={props.on_run.clone()}>{"Run"}</button>
            <button class="editor-btn" onclick={on_load_click}>{"Load"}</button>
            <input ref={file_ref.clone()} type="file" accept=".mlpl,.txt" style="display:none" onchange={on_file_change} />
            <button class="editor-btn" onclick={props.on_save.clone()}>{"Save"}</button>
            <button class="editor-btn" onclick={props.on_clear.clone()}>{"Clear"}</button>
        </div>
    }
}

fn make_input_handler(on_change: Callback<String>) -> Callback<InputEvent> {
    Callback::from(move |e: InputEvent| {
        let ta: HtmlTextAreaElement = e.target_unchecked_into();
        on_change.emit(ta.value());
    })
}

fn make_load_click_handler(file_ref: NodeRef) -> Callback<MouseEvent> {
    Callback::from(move |_: MouseEvent| {
        if let Some(el) = file_ref.cast::<HtmlInputElement>() {
            el.click();
        }
    })
}

fn make_file_change_handler(on_change: Callback<String>) -> Callback<Event> {
    Callback::from(move |e: Event| {
        let input: HtmlInputElement = e.target_unchecked_into();
        let Some(file) = input.files().and_then(|f| f.get(0)) else {
            return;
        };
        let reader = FileReader::new().unwrap();
        let cb = on_change.clone();
        let onload = Closure::wrap(Box::new(move |e: web_sys::ProgressEvent| {
            let r: FileReader = e.target().unwrap().unchecked_into();
            if let Ok(text) = r.result()
                && let Some(s) = text.as_string()
            {
                cb.emit(s);
            }
        }) as Box<dyn FnMut(_)>);
        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        onload.forget();
        let _ = reader.read_as_text(&file);
        input.set_value("");
    })
}

fn make_keydown_handler(on_run: Callback<MouseEvent>) -> Callback<web_sys::KeyboardEvent> {
    Callback::from(move |e: web_sys::KeyboardEvent| {
        if e.ctrl_key() && e.key() == "Enter" {
            e.prevent_default();
            on_run.emit(e.unchecked_into());
        }
    })
}
