use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use web_sys::{HtmlInputElement, KeyboardEvent};
use yew::prelude::*;

use crate::demos::DEMOS;
use crate::help::help_text;
use crate::state::{EntryKind, HistoryEntry};

pub fn toggle_bool(handle: UseStateHandle<bool>, value: bool) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| handle.set(value))
}

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
                kind: EntryKind::Command,
            }
        } else {
            let result = deps.session.borrow().eval(trimmed);
            let is_error = result.starts_with("error:");
            HistoryEntry {
                input: trimmed.to_string(),
                output: result,
                is_error,
                kind: EntryKind::Command,
            }
        };

        new_history.push(entry);
        deps.history.set(new_history);
        deps.cmd_history.set(new_cmds);
        deps.cmd_index.set(None);
        deps.input_value.set(String::new());
    })
}

pub fn make_clear(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        session.borrow().clear();
        history.set(Vec::new());
    })
}

pub fn make_run_demo(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
) -> Callback<usize> {
    Callback::from(move |idx: usize| {
        let Some(demo) = DEMOS.get(idx) else {
            return;
        };
        session.borrow().clear();
        // Lead with a narration panel framing what the demo does
        // and why. Demo lines follow; the takeaway narration lands
        // after the final line in `schedule_demo_line`. Also bind
        // a `_demo` string in the session so `:describe _demo`
        // prints the intro + takeaway from the REPL.
        bind_demo_metadata(&session.borrow(), demo);
        let intro_entry = HistoryEntry {
            input: format!("About this demo -- {}", demo.name),
            output: demo.intro.to_string(),
            is_error: false,
            kind: EntryKind::Narration,
        };
        let entries = vec![intro_entry];
        history.set(entries.clone());
        // Evaluate demo lines asynchronously: each line runs in
        // its own `Timeout::new(0, ...)` tick so the browser can
        // paint the preceding line's output and process input
        // between lines. A long-running single line (e.g. `train
        // 30 { ... }`) still blocks the event loop *during* its
        // own eval -- fixing that needs Web Workers (see
        // docs/worker-threads.md) -- but the cross-line yield
        // keeps the tab from triggering the "unresponsive"
        // dialog on multi-line demos where the total wall clock
        // is the problem.
        //
        // We thread the accumulated `entries` vec through
        // recursion explicitly rather than reading from the
        // `UseStateHandle` inside each tick: a state handle is
        // snapshotted at the callback's closure-creation time
        // and does not refresh between `set()` calls, so reading
        // via `(*history).clone()` inside a deferred Timeout
        // reliably sees the stale initial value and each tick
        // overwrites the previous one's write. Passing `entries`
        // by move keeps a single authoritative Rust-side source
        // of truth; `history.set(entries.clone())` is purely for
        // the UI paint.
        schedule_demo_line(session.clone(), history.clone(), entries, demo, 0);
    })
}

/// Bind `_demo` as a string variable in the session so typing
/// `:describe _demo` after a demo run reprints the intro +
/// takeaway. Uses MLPL's string-assignment syntax through `eval`
/// so the binding goes through the existing string-variable
/// surface (Saga 12 step 009) -- no new plumbing.
fn bind_demo_metadata(session: &WasmSession, demo: &crate::demos::Demo) {
    let body = format!(
        "{}\n\nAbout this demo:\n  {}\n\nTakeaway:\n  {}",
        demo.name, demo.intro, demo.takeaway,
    );
    let escaped = body.replace('\\', "\\\\").replace('"', "\\\"");
    let _ = session.eval(&format!("_demo = \"{escaped}\""));
}

fn schedule_demo_line(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
    mut entries: Vec<HistoryEntry>,
    demo: &'static crate::demos::Demo,
    idx: usize,
) {
    let lines = demo.lines;
    if idx >= lines.len() {
        // Last line ran -- append the takeaway narration panel
        // and paint the final history state.
        entries.push(HistoryEntry {
            input: "What just happened".to_string(),
            output: demo.takeaway.to_string(),
            is_error: false,
            kind: EntryKind::Narration,
        });
        history.set(entries);
        return;
    }
    let session_next = Rc::clone(&session);
    let history_next = history.clone();
    gloo::timers::callback::Timeout::new(0, move || {
        let line = lines[idx];
        let result = session.borrow().eval(line);
        let is_error = result.starts_with("error:");
        entries.push(HistoryEntry {
            input: line.to_string(),
            output: result,
            is_error,
            kind: EntryKind::Command,
        });
        history.set(entries.clone());
        schedule_demo_line(session_next, history_next, entries, demo, idx + 1);
    })
    .forget();
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
