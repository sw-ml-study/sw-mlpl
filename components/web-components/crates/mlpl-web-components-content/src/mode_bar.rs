use web_sys::HtmlSelectElement;
use yew::prelude::*;

use mlpl_web_demos::{DEMOS, Device, capability_for};

/// One dropdown option: `(demo index, name, disabled)`.
type DemoOption = (usize, &'static str, bool);
/// Dropdown sections: `(section label, options)`.
type DemoGroups = Vec<(&'static str, Vec<DemoOption>)>;

/// Group demos for the dropdown by capability tier. `cpu`/live
/// demos keep their authored category; connect-only demos get
/// their own device-tier sections (MLX and CUDA deliberately
/// separate). The bool is `disabled`: connect/GPU demos are
/// visible-but-not-runnable when no server is connected (the
/// public live demo). `connected` comes from connect-mode
/// detection so the same build gates correctly in both modes.
fn grouped_demos(connected: bool) -> DemoGroups {
    let mut map: std::collections::BTreeMap<&str, Vec<DemoOption>> =
        std::collections::BTreeMap::new();
    for (i, d) in DEMOS.iter().enumerate() {
        let cap = capability_for(d.name);
        let section = match cap.device {
            Device::Mlx => "MLX - Apple GPU (connect)",
            Device::Cuda => "CUDA - Linux GPU (connect)",
            Device::Cpu if cap.requires_connect => "Client-server (connect)",
            Device::Cpu => d.category,
        };
        let disabled = cap.requires_connect && !connected;
        map.entry(section).or_default().push((i, d.name, disabled));
    }
    for items in map.values_mut() {
        items.sort_by_key(|(_, name, _)| name.to_ascii_lowercase());
    }
    map.into_iter().collect()
}

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
    let cls = if props.tutorial_active {
        "modebar tutorial"
    } else {
        "modebar repl"
    };
    let demo_dropdown = render_demo_dropdown(props.tutorial_active, props.on_demo.clone());
    let upload_widget = render_upload_widget(props);
    html! {
        <div class={cls}>
            { demo_dropdown }
            { upload_widget }
        </div>
    }
}

fn render_demo_dropdown(tutorial_active: bool, on_demo: Callback<usize>) -> Html {
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
    let groups = grouped_demos(mlpl_web_eval::eval_url::is_connected());
    html! {
        <select class="demo-select" onchange={on_change} aria-label="Load demo" data-tour-target="demo-select">
            <option value="" selected=true>{"Load Demo..."}</option>
            { for groups.iter().map(|(cat, items)| html! {
                <optgroup label={*cat}>
                    { for items.iter().map(|(i, name, disabled)| html! {
                        <option value={i.to_string()} disabled={*disabled}>{ *name }</option>
                    }) }
                </optgroup>
            }) }
        </select>
    }
}

fn render_upload_widget(props: &ModeBarProps) -> Html {
    if props.tutorial_active {
        return html! {};
    }
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
