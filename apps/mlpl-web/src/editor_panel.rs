use web_sys::HtmlTextAreaElement;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct EditorProps {
    pub content: String,
    pub on_change: Callback<String>,
    pub on_run: Callback<MouseEvent>,
    pub on_load: Callback<MouseEvent>,
    pub on_save: Callback<MouseEvent>,
    pub on_clear: Callback<MouseEvent>,
}

#[function_component(EditorPanel)]
pub fn editor_panel(props: &EditorProps) -> Html {
    let on_input = {
        let cb = props.on_change.clone();
        Callback::from(move |e: InputEvent| {
            let ta: HtmlTextAreaElement = e.target_unchecked_into();
            cb.emit(ta.value());
        })
    };
    html! {
        <div class="editor-panel">
            <div class="editor-toolbar">
                <button class="editor-btn editor-btn-run" onclick={props.on_run.clone()}>{"Run"}</button>
                <button class="editor-btn" onclick={props.on_load.clone()}>{"Load"}</button>
                <button class="editor-btn" onclick={props.on_save.clone()}>{"Save"}</button>
                <button class="editor-btn" onclick={props.on_clear.clone()}>{"Clear"}</button>
            </div>
            <textarea
                class="editor-textarea"
                spellcheck="false"
                placeholder="# Type or paste MLPL script here\nx = iota(10)\nreshape(x, [2, 5])"
                value={props.content.clone()}
                oninput={on_input}
            />
        </div>
    }
}
