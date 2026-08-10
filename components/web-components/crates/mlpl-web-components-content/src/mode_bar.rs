use yew::prelude::*;

use crate::demo_gating::{DemoGroups, disabled_hint, grouped_demos, reachable_connected};
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
    let picker = (!props.tutorial_active).then(|| {
        html! { <DemoDropdown on_demo={props.on_demo.clone()} peer_devices={probe.devices.clone()} /> }
    });
    let upload = (!props.tutorial_active).then(|| render_upload_widget(props));
    html! {
        <div class={cls}>
            { banner }
            { picker }
            { upload }
        </div>
    }
}

#[derive(Properties, PartialEq)]
struct DemoDropdownProps {
    on_demo: Callback<usize>,
    peer_devices: Vec<Device>,
}

/// A custom (non-native) demo picker. The native `<select>` /
/// `<optgroup>` cannot be styled reliably across browsers (Safari
/// ignores optgroup styling entirely; Chrome ignores its
/// font-size), so the demo-GROUP names rendered as low-contrast
/// muted text. This renders a toggle button plus a panel whose
/// group headers and rows are ordinary styled elements, legible in
/// every browser. Capability gating and ordering are unchanged --
/// both still come from `grouped_demos`.
#[function_component(DemoDropdown)]
fn demo_dropdown(props: &DemoDropdownProps) -> Html {
    let open = use_state(|| false);
    let groups = grouped_demos(reachable_connected(), &props.peer_devices);
    let toggle = {
        let open = open.clone();
        Callback::from(move |_: MouseEvent| open.set(!*open))
    };
    let pick = {
        let open = open.clone();
        let on_demo = props.on_demo.clone();
        Callback::from(move |i: usize| {
            on_demo.emit(i);
            open.set(false);
        })
    };
    let panel = open.then(|| render_panel(&groups, &pick, &toggle));
    html! {
        <div class="demo-dropdown">
            <button class="demo-select" onclick={toggle.clone()} aria-haspopup="listbox" aria-expanded={open.to_string()} aria-label="Load demo" data-tour-target="demo-select">{"Load Demo..."}</button>
            { panel }
        </div>
    }
}

/// The open panel: a full-viewport backdrop that closes on click,
/// then one styled group header + gated demo rows per section. Each
/// row is a button that emits its index (or a disabled row carrying
/// the capability hint -- identical gating to the old
/// `<option disabled>`).
fn render_panel(
    groups: &DemoGroups,
    pick: &Callback<usize>,
    toggle: &Callback<MouseEvent>,
) -> Html {
    html! {
        <>
            <div class="demo-dropdown-backdrop" onclick={toggle.clone()} />
            <div class="demo-dropdown-panel" role="listbox" aria-label="Demos">
                { for groups.iter().map(|(cat, items)| html! {
                    <div class="demo-group">
                        <div class="demo-group-label">{ *cat }</div>
                        { for items.iter().map(|(i, name, disabled)| {
                            let idx = *i;
                            let pick = pick.clone();
                            let onclick = Callback::from(move |_: MouseEvent| pick.emit(idx));
                            let title = disabled.then(|| AttrValue::from(disabled_hint(cat)));
                            html! { <button class="demo-item" role="option" disabled={*disabled} title={title} onclick={onclick}>{ name.to_string() }</button> }
                        }) }
                    </div>
                }) }
            </div>
        </>
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
