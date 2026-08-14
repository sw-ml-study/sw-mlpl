use yew::prelude::*;

use crate::demo_gating::{
    DemoGroups, DemoOption, GROUP_TOOLTIPS, demo_tooltip, grouped_demos, reachable_connected,
};
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
    // Which top-level super-group is expanded (accordion; at most one).
    let expanded = use_state(|| Option::<usize>::None);
    let supers = super_grouped(grouped_demos(reachable_connected(), &props.peer_devices));
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
    let panel = open.then(|| render_super_panel(&supers, &expanded, &pick, &toggle));
    html! {
        <div class="demo-dropdown">
            <button class="demo-select" onclick={toggle.clone()} aria-haspopup="menu" aria-expanded={open.to_string()} aria-label="Load demo" data-tour-target="demo-select">{"Load Demo..."}</button>
            { panel }
        </div>
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

// ----- Nested demo menu: subject SUPER-GROUPS over the flat category
// sections, so the picker isn't one very long list. Kept in this file
// (rather than sibling modules) because the crate is at its module
// budget; the crate wants splitting (queued tech debt).

/// Nested menu: `(super-group label, its category sections)`.
type SuperGrouped = Vec<(&'static str, DemoGroups)>;

/// Each super-group and the category sections it collects, in menu
/// order. A super-group with no present categories is dropped (e.g.
/// Mathematics, until the abstract-algebra / category-theory demos
/// land). Unlisted categories fall into a trailing "Other".
const SUPER_GROUPS: &[(&str, &[&str])] = &[
    (
        "Machine Learning",
        &[
            "Training & Learning",
            "Experiment Quality",
            "Classical ML",
            "Classification",
            "Clustering",
            "Dim Reduction",
            "Vision",
            "Attention",
            "Sequence Models",
            "Language Models",
            "Generative Models",
            "Engram",
        ],
    ),
    ("Mathematics", &["Abstract Algebra", "Category Theory"]),
    ("Array / APL", &["Basics", "APL2 / General Programming"]),
    (
        "Systems & Tooling",
        &[
            "Data Forge",
            "Non-Browser (companion CLIs)",
            "Client-server (connect)",
            "MLX - Apple GPU (connect)",
            "CUDA - Linux GPU (connect)",
        ],
    ),
];

/// Fold flat category `groups` into ordered super-groups (categories
/// are moved into the first super-group that claims them; the rest
/// become a trailing "Other"; empty super-groups vanish).
fn super_grouped(mut groups: DemoGroups) -> SuperGrouped {
    let mut out: SuperGrouped = Vec::new();
    for (label, cats) in SUPER_GROUPS.iter() {
        let mut section: DemoGroups = Vec::new();
        for cat in cats.iter() {
            if let Some(pos) = groups.iter().position(|(c, _)| c == cat) {
                section.push(groups.remove(pos));
            }
        }
        if !section.is_empty() {
            out.push((*label, section));
        }
    }
    if !groups.is_empty() {
        out.push(("Other", groups));
    }
    out
}

/// The open panel: a backdrop that closes on click, then one
/// collapsible super-group per section (accordion: at most one open).
fn render_super_panel(
    supers: &SuperGrouped,
    expanded: &UseStateHandle<Option<usize>>,
    pick: &Callback<usize>,
    toggle: &Callback<MouseEvent>,
) -> Html {
    html! {
        <>
            <div class="demo-dropdown-backdrop" onclick={toggle.clone()} />
            <div class="demo-dropdown-panel" role="menu" aria-label="Demos">
                { for supers.iter().enumerate().map(|(i, (label, cats))| {
                    render_super_group(i, label, cats, expanded, pick)
                }) }
            </div>
        </>
    }
}

/// One super-group: a header (name + demo count) that expands/collapses
/// its category sections.
fn render_super_group(
    idx: usize,
    label: &str,
    cats: &DemoGroups,
    expanded: &UseStateHandle<Option<usize>>,
    pick: &Callback<usize>,
) -> Html {
    let is_open = **expanded == Some(idx);
    let on_click = {
        let expanded = expanded.clone();
        Callback::from(move |_: MouseEvent| {
            expanded.set(if is_open { None } else { Some(idx) });
        })
    };
    let count: usize = cats.iter().map(|(_, items)| items.len()).sum();
    let body = is_open.then(|| {
        html! {
            <div class="demo-super-body">
                { for cats.iter().map(|(cat, items)| render_category(cat, items, pick)) }
            </div>
        }
    });
    html! {
        <div class="demo-super">
            <button class="demo-super-header" onclick={on_click} aria-expanded={is_open.to_string()}>
                <span class="demo-super-caret" aria-hidden="true">{ if is_open { "\u{25be}" } else { "\u{25b8}" } }</span>
                <span class="demo-super-label">{ label.to_string() }</span>
                <span class="demo-super-count">{ count.to_string() }</span>
            </button>
            { body }
        </div>
    }
}

/// One category section inside an expanded super-group: a group label
/// (with tooltip) plus its gated demo rows.
fn render_category(cat: &str, items: &[DemoOption], pick: &Callback<usize>) -> Html {
    let tip = GROUP_TOOLTIPS.iter().find(|(name, _)| *name == cat).map_or(
        "Related MLPL demonstrations grouped by subject.",
        |(_, t)| *t,
    );
    html! {
        <div class="demo-group">
            <div class="demo-group-label demo-tooltip-target" tabindex="0" data-tooltip={tip.to_string()} aria-label={format!("{cat}: {tip}")}>{ cat.to_string() }</div>
            { for items.iter().map(|(i, name, disabled)| {
                let idx = *i;
                let pick = pick.clone();
                let onclick = Callback::from(move |_: MouseEvent| pick.emit(idx));
                let title = AttrValue::from(demo_tooltip(idx, cat, *disabled));
                html! {
                    <span class="demo-tooltip-target" tabindex={if *disabled { "0" } else { "-1" }} data-tooltip={title.clone()}>
                        <button class="demo-item" role="menuitem" disabled={*disabled} aria-label={title} onclick={onclick}>{ name.to_string() }</button>
                    </span>
                }
            }) }
        </div>
    }
}
