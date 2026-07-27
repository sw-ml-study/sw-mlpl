use web_sys::HtmlSelectElement;
use yew::prelude::*;

use crate::demo_gating::{disabled_hint, grouped_demos};
use crate::peer_probe::{connect_banner, use_peer_probe};
use mlpl_web_demos::Device;

#[derive(Properties, PartialEq)]
pub struct ModeBarProps {
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
    let mode = if props.tutorial_active {
        "tutorial"
    } else {
        "repl"
    };
    let cls = format!("modebar {mode}");
    // Probe the connected peer's real device set, so the dropdown gates
    // GPU demos by what THIS peer offers (CUDA on a Linux peer, MLX on
    // an Apple peer) -- not a static guess. The banner surfaces an
    // invalid ?connect= or an unresponsive server loudly at load.
    let probe = use_peer_probe();
    let banner = connect_banner(&probe);
    let demo_dropdown =
        render_demo_dropdown(props.tutorial_active, props.on_demo.clone(), &probe.devices);
    let upload = (!props.tutorial_active).then(|| render_upload_widget(props));
    html! {
        <div class={cls}>
            { banner }
            { demo_dropdown }
            { upload }
        </div>
    }
}

fn render_demo_dropdown(
    tutorial_active: bool,
    on_demo: Callback<usize>,
    peer_devices: &[Device],
) -> Html {
    if tutorial_active {
        return html! {};
    }
    let on_change = Callback::from(move |e: Event| {
        let target: HtmlSelectElement = e.target_unchecked_into();
        if let Ok(i) = target.value().parse::<usize>() {
            on_demo.emit(i);
            target.set_value("");
        }
    });
    let groups = grouped_demos(crate::demo_gating::reachable_connected(), peer_devices);
    html! {
        <select class="demo-select" onchange={on_change} aria-label="Load demo" data-tour-target="demo-select">
            <option value="" selected=true>{"Load Demo..."}</option>
            { for groups.iter().map(|(cat, items)| demo_optgroup(cat, items)) }
        </select>
    }
}

/// One dropdown section: the category label plus its (possibly
/// gated) demo options, alphabetical within the group.
fn demo_optgroup(cat: &'static str, items: &[crate::demo_gating::DemoOption]) -> Html {
    html! {
        <optgroup label={cat} title={cat.to_string()}>
            { for items.iter().map(|(i, name, disabled)| {
                let hint = disabled.then(|| AttrValue::from(disabled_hint(cat)));
                html! {
                    <option value={i.to_string()} disabled={*disabled} title={hint}>{ *name }</option>
                }
            }) }
        </optgroup>
    }
}

fn render_upload_widget(props: &ModeBarProps) -> Html {
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
}
