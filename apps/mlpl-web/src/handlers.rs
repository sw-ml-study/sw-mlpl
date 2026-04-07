use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use web_sys::{HtmlInputElement, KeyboardEvent};
use yew::prelude::*;

use crate::help::help_text;
use crate::state::HistoryEntry;

pub struct EvalDeps {
    pub session: Rc<RefCell<WasmSession>>,
    pub history: UseStateHandle<Vec<HistoryEntry>>,
    pub input_value: UseStateHandle<String>,
    pub cmd_history: UseStateHandle<Vec<String>>,
    pub cmd_index: UseStateHandle<Option<usize>>,
}

pub fn make_submit(deps: EvalDeps) -> Callback<String> {
    Callback::from(move |line: String| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }

        let mut new_history = (*deps.history).clone();
        let mut new_cmds = (*deps.cmd_history).clone();
        new_cmds.push(trimmed.to_string());

        if trimmed == ":clear" {
            deps.session.borrow().clear();
            new_history.clear();
            deps.cmd_history.set(new_cmds);
            deps.cmd_index.set(None);
            deps.input_value.set(String::new());
            deps.history.set(new_history);
            return;
        }

        let entry = if trimmed == ":help" {
            HistoryEntry {
                input: trimmed.to_string(),
                output: help_text(),
                is_error: false,
            }
        } else {
            let result = deps.session.borrow().eval(trimmed);
            let is_error = result.starts_with("error:");
            HistoryEntry {
                input: trimmed.to_string(),
                output: result,
                is_error,
            }
        };

        new_history.push(entry);
        deps.history.set(new_history);
        deps.cmd_history.set(new_cmds);
        deps.cmd_index.set(None);
        deps.input_value.set(String::new());
    })
}

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
) -> Callback<KeyboardEvent> {
    Callback::from(move |e: KeyboardEvent| match e.key().as_str() {
        "Enter" => {
            e.prevent_default();
            on_submit.emit((*input_value).clone());
        }
        "ArrowUp" => {
            e.prevent_default();
            let cmds = &*cmd_history;
            if cmds.is_empty() {
                return;
            }
            let new_idx = match *cmd_index {
                None => cmds.len() - 1,
                Some(0) => 0,
                Some(i) => i - 1,
            };
            cmd_index.set(Some(new_idx));
            input_value.set(cmds[new_idx].clone());
        }
        "ArrowDown" => {
            e.prevent_default();
            let cmds = &*cmd_history;
            match *cmd_index {
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
        _ => {}
    })
}
