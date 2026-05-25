use web_sys::HtmlSelectElement;
use yew::prelude::*;

use crate::demos::DEMOS;

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
            <select class="demo-select" onchange={on_change} aria-label="Load demo" data-tour-target="demo-select">
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
