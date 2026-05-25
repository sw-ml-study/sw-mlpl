use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use yew::prelude::*;

use crate::demos::DEMOS;
use crate::handlers_submit::{push_running_marker, replace_running_with_result};
use mlpl_web_eval::state::{EntryKind, HistoryEntry};

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
