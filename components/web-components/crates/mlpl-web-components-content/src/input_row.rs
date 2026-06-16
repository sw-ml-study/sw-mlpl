use web_sys::KeyboardEvent;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct InputRowProps {
    pub value: String,
    pub on_input: Callback<InputEvent>,
    pub on_keydown: Callback<KeyboardEvent>,
    pub in_tutorial: bool,
    pub completion_candidates: Vec<String>,
    pub on_pick_completion: Callback<String>,
    /// Saga 33 step 047: highlighted chip index.
    pub completion_selected: usize,
    /// Saga 83: contextual Reset callback. Renders next to the
    /// input as "Reset Tutorial" when `in_tutorial`, "Reset REPL"
    /// otherwise. The button used to live in the top mode bar
    /// but was too far from where output appeared; moving it
    /// next to the prompt makes the affordance obvious.
    pub on_clear: Callback<MouseEvent>,
    /// Whether the 3D stage is currently shown. Drives which
    /// half of the `2D | 3D` toggle is bold (active).
    pub show_3d: bool,
    /// Flip 2D <-> 3D. A visual convenience for the `:2d` /
    /// `:3d` commands; both drive the same global `show_3d`
    /// state, so the button, the commands, and Ctrl+3 stay in
    /// sync. Lives on every REPL input line (main + tutorial).
    pub on_toggle_3d: Callback<MouseEvent>,
}

/// Render the `2D | 3D` segmented control. The active mode is
/// bold and inert; the other mode is a link-styled button that
/// flips the view. Clicking the active half is a no-op by
/// design -- the user clicks the half they want to switch TO.
fn render_mode_toggle(show_3d: bool, on_toggle: &Callback<MouseEvent>) -> Html {
    let title = "Switch 2D / 3D view (same as the :2d and :3d commands)";
    let opt = |is_active: bool, label: &'static str| -> Html {
        if is_active {
            html! { <span class="mode-opt active">{ label }</span> }
        } else {
            html! { <button class="mode-opt" onclick={on_toggle.clone()}>{ label }</button> }
        }
    };
    html! {
        <span class="mode-toggle" title={title}>
            { opt(!show_3d, "2D") }<span class="mode-sep">{"|"}</span>{ opt(show_3d, "3D") }
        </span>
    }
}

/// Render the completion-candidate popup. Empty when there are no
/// candidates; otherwise one clickable chip per candidate, with the
/// selected chip highlighted and announced via `aria-selected`.
fn render_completion_popup(candidates: &[String], sel: usize, on_pick: &Callback<String>) -> Html {
    if candidates.is_empty() {
        return html! {};
    }
    let chips = candidates.iter().enumerate().map(|(i, c)| {
        let cb = on_pick.clone();
        let text = c.clone();
        let onclick = Callback::from(move |_| cb.emit(text.clone()));
        let cls = if i == sel { "completion-chip selected" } else { "completion-chip" };
        html! {
            <button class={cls} onclick={onclick} title="Click to insert" role="option" aria-selected={if i == sel { "true" } else { "false" }}>{ c.clone() }</button>
        }
    });
    html! {
        <div class="completion-popup" role="listbox" aria-label="Completion candidates" data-tour-target="completion-popup">
            { for chips }
        </div>
    }
}

#[function_component(InputRow)]
pub fn input_row(props: &InputRowProps) -> Html {
    let (label_text, label_class) = if props.in_tutorial {
        ("(Tutorial)", "session-label tutorial")
    } else {
        ("(REPL)", "session-label repl")
    };
    let popup = render_completion_popup(
        &props.completion_candidates,
        props.completion_selected,
        &props.on_pick_completion,
    );
    let clear_label = if props.in_tutorial {
        "Reset Tutorial"
    } else {
        "Reset REPL"
    };
    html! {
        <div class="input-wrap">
            <div class={label_class}>{ label_text }</div>
            <div class="input-row">
                <span class="prompt">{"mlpl> "}</span>
                <input
                    id="repl-input"
                    data-tour-target="repl-input"
                    type="text"
                    autocomplete="off"
                    spellcheck="false"
                    value={props.value.clone()}
                    oninput={props.on_input.clone()}
                    onkeydown={props.on_keydown.clone()}
                />
                { render_mode_toggle(props.show_3d, &props.on_toggle_3d) }
                <button
                    class="ctrl-btn input-row-clear"
                    onclick={props.on_clear.clone()}
                    title="Clear all output from this session"
                >
                    { clear_label }
                </button>
            </div>
            { popup }
        </div>
    }
}
