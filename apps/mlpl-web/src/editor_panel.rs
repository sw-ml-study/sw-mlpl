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
    let on_input = {
        let cb = props.on_change.clone();
        Callback::from(move |e: InputEvent| {
            let ta: HtmlTextAreaElement = e.target_unchecked_into();
            cb.emit(ta.value());
        })
    };
    let on_load_click = {
        let r = file_ref.clone();
        Callback::from(move |_: MouseEvent| {
            if let Some(el) = r.cast::<HtmlInputElement>() {
                el.click();
            }
        })
    };
    let on_file_change = {
        let cb = props.on_change.clone();
        Callback::from(move |e: Event| {
            let input: HtmlInputElement = e.target_unchecked_into();
            let Some(file) = input.files().and_then(|f| f.get(0)) else {
                return;
            };
            let reader = FileReader::new().unwrap();
            let cb2 = cb.clone();
            let onload = Closure::wrap(Box::new(move |e: web_sys::ProgressEvent| {
                let reader: FileReader = e.target().unwrap().unchecked_into();
                if let Ok(text) = reader.result()
                    && let Some(s) = text.as_string()
                {
                    cb2.emit(s);
                }
            }) as Box<dyn FnMut(_)>);
            reader.set_onload(Some(onload.as_ref().unchecked_ref()));
            onload.forget();
            let _ = reader.read_as_text(&file);
            input.set_value("");
        })
    };
    let on_keydown = {
        let run = props.on_run.clone();
        Callback::from(move |e: web_sys::KeyboardEvent| {
            if e.ctrl_key() && e.key() == "Enter" {
                e.prevent_default();
                run.emit(e.unchecked_into());
            }
        })
    };
    html! {
        <div class="editor-panel">
            <div class="editor-toolbar">
                <button class="editor-btn editor-btn-run" onclick={props.on_run.clone()}>{"Run"}</button>
                <button class="editor-btn" onclick={on_load_click}>{"Load"}</button>
                <input ref={file_ref} type="file" accept=".mlpl,.txt" style="display:none" onchange={on_file_change} />
                <button class="editor-btn" onclick={props.on_save.clone()}>{"Save"}</button>
                <button class="editor-btn" onclick={props.on_clear.clone()}>{"Clear"}</button>
            </div>
            <textarea
                class="editor-textarea"
                spellcheck="false"
                placeholder={"# Type or paste MLPL script here\nx = iota(10)\nreshape(x, [2, 5])"}
                value={props.content.clone()}
                oninput={on_input}
                onkeydown={on_keydown}
            />
        </div>
    }
}
