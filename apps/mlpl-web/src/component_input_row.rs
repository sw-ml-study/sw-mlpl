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
}

#[function_component(InputRow)]
pub fn input_row(props: &InputRowProps) -> Html {
    let (label_text, label_class) = if props.in_tutorial {
        ("(Tutorial)", "session-label tutorial")
    } else {
        ("(REPL)", "session-label repl")
    };
    let popup = if props.completion_candidates.is_empty() {
        html! {}
    } else {
        let sel = props.completion_selected;
        let chips = props.completion_candidates.iter().enumerate().map(|(i, c)| {
            let cb = props.on_pick_completion.clone();
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
            </div>
            { popup }
        </div>
    }
}
