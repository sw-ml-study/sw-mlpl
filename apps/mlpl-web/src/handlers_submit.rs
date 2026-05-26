use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use yew::prelude::*;

use crate::help::help_text;
use crate::upload::PendingUploadName;
use crate::upload_cmd::{handle_upload_command, parse_upload_command};
use mlpl_web_eval::state::{EntryKind, HistoryEntry};

#[derive(Clone)]
pub struct EvalDeps {
    pub session: Rc<RefCell<WasmSession>>,
    pub history: UseStateHandle<Vec<HistoryEntry>>,
    pub input_value: UseStateHandle<String>,
    pub cmd_history: UseStateHandle<Vec<String>>,
    pub cmd_index: UseStateHandle<Option<usize>>,
    /// Saga 29 step 016: noderef + pending-name handle for the
    /// `:upload <name>` REPL command. `upload_input_ref` points
    /// at the hidden `<input type=file>` rendered by the REPL
    /// controls; `pending_upload_name` is set to `Some(<name>)`
    /// before the file picker fires, so the existing on-change
    /// pipeline knows which session variable to bind.
    pub upload_input_ref: NodeRef,
    pub show_3d: UseStateHandle<bool>,
    pub pending_upload_name: PendingUploadName,
}

/// Single-line submit: thin wrapper that pipes the line through
/// the batch path so there is exactly one place that mutates
/// state. Calling `make_submit` N times in a tight loop USED
/// to drop N-1 history entries because each call did
/// `(*deps.history).clone()` -> push one -> `set(new)` --
/// inside one event tick every clone read the same snapshot
/// and only the last `set` won (Tutorial Run-All bug, May 7
/// 2026). Routing through `make_submit_batch` collapses any
/// number of lines into a single read-modify-write.
pub fn make_submit(deps: EvalDeps) -> Callback<String> {
    let batch = make_submit_batch(deps);
    Callback::from(move |line: String| batch.emit(vec![line]))
}

/// Evaluate every line in `lines` against the current session
/// in one pass, then commit all four state handles exactly
/// once. Empty/whitespace-only lines are skipped (matching
/// single-line submit semantics); `:clear` and `:help` are
/// honored mid-batch.
pub fn make_submit_batch(deps: EvalDeps) -> Callback<Vec<String>> {
    Callback::from(move |lines: Vec<String>| {
        let mut new_history = (*deps.history).clone();
        let mut new_cmds = (*deps.cmd_history).clone();
        let mut eval_queue: Vec<String> = Vec::new();
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            new_cmds.push(trimmed.to_string());
            if trimmed == ":clear" {
                deps.session.borrow().clear();
                new_history.clear();
                if *deps.show_3d {
                    let _ = js_sys::eval("window.__stage3d_clear && window.__stage3d_clear()");
                }
                continue;
            }
            if let Some(explicit) = crate::viz3d_toggle::parse_3d_command(trimmed) {
                deps.show_3d.set(explicit.unwrap_or(!*deps.show_3d));
                continue;
            }
            if let Some(name) = parse_upload_command(trimmed) {
                new_history.push(handle_upload_command(&deps, trimmed, &name));
                continue;
            }
            // Anything else hits the WASM evaluator, which may
            // block for seconds; defer to the spinner pipeline
            // (Saga 29 step 018) so the user sees a running
            // indicator instead of a frozen page.
            eval_queue.push(trimmed.to_string());
        }
        if new_cmds.is_empty() {
            return;
        }
        deps.cmd_history.set(new_cmds);
        deps.cmd_index.set(None);
        deps.input_value.set(String::new());
        if eval_queue.is_empty() {
            deps.history.set(new_history);
            return;
        }
        // Saga 29 step 018: push Running markers for each
        // queued line + paint, then process them one-at-a-time
        // through Timeout(0) yields so the spinner CSS keeps
        // animating during each eval.
        process_next_eval(deps.clone(), new_history, eval_queue, 0);
    })
}

/// Saga 29 step 018: recursive Timeout-yielding eval loop for
/// REPL submissions. Pushes a Running marker, yields, evals,
/// replaces marker with result, recurses to next line.
fn process_next_eval(
    deps: EvalDeps,
    mut history: Vec<HistoryEntry>,
    queue: Vec<String>,
    idx: usize,
) {
    if idx >= queue.len() {
        deps.history.set(history);
        return;
    }
    let line = queue[idx].clone();
    push_running_marker(&mut history, &line);
    deps.history.set(history.clone());
    let deps_next = deps.clone();
    let queue_next = queue.clone();
    gloo::timers::callback::Timeout::new(0, move || {
        let entry = eval_one_line(&deps_next, &line);
        if *deps_next.show_3d && !entry.is_error && !line.starts_with(':') {
            let (shape, elements) = crate::viz3d_events::shape_from_output(&entry.output);
            crate::viz3d_events::emit(&crate::viz3d_events::Stage3dEvent {
                step_idx: 0,
                label: line.clone(),
                output: crate::viz3d_events::ShapeInfo {
                    name: line.split('=').next().unwrap_or(&line).trim().to_string(),
                    shape: shape.clone(),
                    rank: shape.len(),
                    elements,
                },
            });
        }
        history.pop();
        history.push(entry);
        process_next_eval(deps_next, history, queue_next, idx + 1);
    })
    .forget();
}

/// Saga 29 step 018: human-readable "this line is running"
/// caption tailored to the kind of line. Train blocks call out
/// the expected duration; everything else gets a generic
/// "evaluating..." marker that still tells the user the demo
/// hasn't hung. Saga 33 step 027: a "train 5"-shaped chunk
/// gets a chunk-sized caption so the user understands each
/// chunk is short and the page will recover between chunks;
/// large `train 30`+ blocks keep the long-duration warning.
pub(crate) fn running_message(line: &str) -> &'static str {
    let stripped = line.trim_start();
    if stripped.starts_with("train ") || stripped.starts_with("train{") {
        return train_caption(stripped);
    }
    if stripped.starts_with("repeat ") {
        "looping... (this can take a few seconds)"
    } else if stripped.contains("predict_batch")
        || stripped.contains("apply(")
        || stripped.contains("attention_weights")
    {
        "evaluating... (forward pass through the trained model)"
    } else if stripped.contains("svg(") {
        "rendering visualization..."
    } else {
        "evaluating..."
    }
}

fn train_caption(stripped: &str) -> &'static str {
    // Parse the iteration count out of `train N {...}`. Small
    // chunks (<=10) get a chunk-shaped message ("the page will
    // un-freeze between chunks"); larger blocks keep the
    // long-duration warning.
    let after_train = stripped["train ".len()..].trim_start();
    let count: usize = after_train
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap_or("")
        .parse()
        .unwrap_or(0);
    if count > 0 && count <= 10 {
        "training chunk... (a few seconds; the page un-freezes between chunks)"
    } else {
        "training... (this can take 30-90 seconds; the page is unresponsive while WASM runs)"
    }
}

/// Evaluate one non-slash-command line and return the
/// HistoryEntry. Handles `:help` inline; everything else
/// hits `session.eval`. Pulled out so `make_submit_batch`
/// stays under the 50-line per-function budget.
fn eval_one_line(deps: &EvalDeps, trimmed: &str) -> HistoryEntry {
    if trimmed == ":help" {
        return HistoryEntry {
            input: trimmed.to_string(),
            output: help_text(),
            is_error: false,
            kind: EntryKind::Command,
        };
    }
    let result = deps.session.borrow().eval(trimmed);
    let is_error = result.starts_with("error:");
    HistoryEntry {
        input: trimmed.to_string(),
        output: result,
        is_error,
        kind: EntryKind::Command,
    }
}

/// Saga 29 step 018: push a Running placeholder so the browser
/// paints a spinner before the (blocking) WASM eval starts.
pub(crate) fn push_running_marker(entries: &mut Vec<HistoryEntry>, line: &str) {
    entries.push(HistoryEntry {
        input: line.to_string(),
        output: running_message(line).to_string(),
        is_error: false,
        kind: EntryKind::Running,
    });
}

/// Saga 29 step 018: pop the trailing Running marker and push
/// the actual Command result. Called from inside the
/// post-eval Timeout closure once the WASM session returns.
pub(crate) fn replace_running_with_result(
    entries: &mut Vec<HistoryEntry>,
    line: &str,
    result: String,
) {
    let is_error = result.starts_with("error:");
    entries.pop();
    entries.push(HistoryEntry {
        input: line.to_string(),
        output: result,
        is_error,
        kind: EntryKind::Command,
    });
}
