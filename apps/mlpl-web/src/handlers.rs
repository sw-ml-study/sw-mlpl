use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use wasm_bindgen::JsCast;
use web_sys::{HtmlInputElement, KeyboardEvent};
use yew::prelude::*;

use crate::demos::DEMOS;
use crate::help::help_text;
use crate::upload::PendingUploadName;
use mlpl_web_eval::state::{EntryKind, HistoryEntry};

pub fn toggle_bool(handle: UseStateHandle<bool>, value: bool) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| handle.set(value))
}

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

// `:upload`-related helpers live in `crate::upload_cmd`
// (separate file) so handlers.rs stays under the 500-line
// budget after step 018's spinner pipeline addition.
use crate::upload_cmd::{handle_upload_command, parse_upload_command};

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

fn push_progress_notes(entries: &mut Vec<HistoryEntry>, demo_name: &str, idx: usize) -> bool {
    let mut had = false;
    for note in crate::demos::progress_notes_for(demo_name, idx) {
        entries.push(HistoryEntry {
            input: note.heading.to_string(),
            output: note.body.to_string(),
            is_error: false,
            kind: EntryKind::Narration,
        });
        had = true;
    }
    had
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
    push_progress_notes(&mut entries, demo.name, idx);
    let line = lines[idx];
    push_running_marker(&mut entries, line);
    history.set(entries.clone());
    let session_next = Rc::clone(&session);
    let history_next = history.clone();
    gloo::timers::callback::Timeout::new(0, move || {
        let result = session.borrow().eval(line);
        replace_running_with_result(&mut entries, line, result);
        history.set(entries.clone());
        schedule_demo_line(session_next, history_next, entries, demo, idx + 1);
    })
    .forget();
}

/// Saga 29 step 018: push a Running placeholder so the browser
/// paints a spinner before the (blocking) WASM eval starts.
fn push_running_marker(entries: &mut Vec<HistoryEntry>, line: &str) {
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
fn replace_running_with_result(entries: &mut Vec<HistoryEntry>, line: &str, result: String) {
    let is_error = result.starts_with("error:");
    entries.pop();
    entries.push(HistoryEntry {
        input: line.to_string(),
        output: result,
        is_error,
        kind: EntryKind::Command,
    });
}

// `running_message` lives earlier in this module as
// `pub(crate)` so both the demo runner and the REPL submit
// pipeline can call it. (A second copy that used to live
// here was removed in step 018.)

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
        if crate::handlers_popup::handle_completion_keys(
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

#[cfg(test)]
mod tests {
    use crate::upload_cmd::{is_valid_identifier, parse_upload_command};

    #[test]
    fn parse_upload_with_name() {
        assert_eq!(parse_upload_command(":upload x"), Some("x".into()));
        assert_eq!(
            parse_upload_command(":upload my_photo"),
            Some("my_photo".into())
        );
        assert_eq!(
            parse_upload_command(":upload   spaced"),
            Some("spaced".into())
        );
    }

    #[test]
    fn parse_upload_no_name_returns_empty_string() {
        assert_eq!(parse_upload_command(":upload"), Some(String::new()));
        assert_eq!(parse_upload_command(":upload   "), Some(String::new()));
    }

    #[test]
    fn parse_non_upload_returns_none() {
        assert_eq!(parse_upload_command(":help"), None);
        assert_eq!(parse_upload_command(":vars"), None);
        assert_eq!(parse_upload_command("x = 1"), None);
        assert_eq!(parse_upload_command(""), None);
        // `:uploaded` shares a prefix with `:upload` but is a
        // different command (no separator); today the parser
        // treats it as `:upload <name>` = `:upload ed`. This is
        // a known quirk -- nobody types `:uploaded` so it
        // doesn't collide in practice.
    }

    #[test]
    fn identifier_validation() {
        assert!(is_valid_identifier("x"));
        assert!(is_valid_identifier("X"));
        assert!(is_valid_identifier("my_photo"));
        assert!(is_valid_identifier("_x"));
        assert!(is_valid_identifier("a1"));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("1"));
        assert!(!is_valid_identifier("1x"));
        assert!(!is_valid_identifier("x y"));
        assert!(!is_valid_identifier("x-y"));
        assert!(!is_valid_identifier("x.y"));
    }
}
