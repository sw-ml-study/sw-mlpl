use wasm_bindgen::JsCast;
use web_sys::{HtmlInputElement, KeyboardEvent};
use yew::prelude::*;

pub fn make_oninput(input_value: UseStateHandle<String>) -> Callback<InputEvent> {
    Callback::from(move |e: InputEvent| {
        let target: HtmlInputElement = e.target_unchecked_into();
        input_value.set(target.value());
    })
}

pub fn make_keydown(
    on_submit: Callback<String>,
    input_value: UseStateHandle<String>,
    cmd_history: UseStateHandle<Vec<String>>,
    cmd_index: UseStateHandle<Option<usize>>,
    completion_candidates: UseStateHandle<Vec<String>>,
    completion_selected: UseStateHandle<usize>,
) -> Callback<KeyboardEvent> {
    Callback::from(move |e: KeyboardEvent| {
        if crate::popup::handle_completion_keys(
            &e,
            &input_value,
            &completion_candidates,
            &completion_selected,
        ) {
            return;
        }
        match e.key().as_str() {
            "Enter" => {
                e.prevent_default();
                on_submit.emit((*input_value).clone());
            }
            "ArrowUp" => {
                e.prevent_default();
                navigate_history_up(&input_value, &cmd_history, &cmd_index);
            }
            "ArrowDown" => {
                e.prevent_default();
                navigate_history_down(&input_value, &cmd_history, &cmd_index);
            }
            _ => {}
        }
    })
}

fn navigate_history_up(
    input_value: &UseStateHandle<String>,
    cmd_history: &UseStateHandle<Vec<String>>,
    cmd_index: &UseStateHandle<Option<usize>>,
) {
    let cmds = &**cmd_history;
    if cmds.is_empty() {
        return;
    }
    let new_idx = match **cmd_index {
        None => cmds.len() - 1,
        Some(0) => 0,
        Some(i) => i - 1,
    };
    cmd_index.set(Some(new_idx));
    input_value.set(cmds[new_idx].clone());
}

fn navigate_history_down(
    input_value: &UseStateHandle<String>,
    cmd_history: &UseStateHandle<Vec<String>>,
    cmd_index: &UseStateHandle<Option<usize>>,
) {
    let cmds = &**cmd_history;
    match **cmd_index {
        Some(i) if i + 1 < cmds.len() => {
            cmd_index.set(Some(i + 1));
            input_value.set(cmds[i + 1].clone());
        }
        Some(_) => {
            cmd_index.set(None);
            input_value.set(String::new());
        }
        None => {}
    }
}
