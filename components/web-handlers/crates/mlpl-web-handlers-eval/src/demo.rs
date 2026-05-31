use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use yew::prelude::*;

use crate::running::{push_running_marker, replace_running_with_result};
use mlpl_web_demos::DEMOS;
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
        let _ = js_sys::eval("window.__stage3d_clear && window.__stage3d_clear()");
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
fn bind_demo_metadata(session: &WasmSession, demo: &mlpl_web_demos::Demo) {
    let body = format!(
        "{}\n\nAbout this demo:\n  {}\n\nTakeaway:\n  {}",
        demo.name, demo.intro, demo.takeaway,
    );
    let escaped = body.replace('\\', "\\\\").replace('"', "\\\"");
    let _ = session.eval(&format!("_demo = \"{escaped}\""));
}

fn push_progress_notes(entries: &mut Vec<HistoryEntry>, demo_name: &str, idx: usize) -> bool {
    let mut had = false;
    for note in mlpl_web_demos::progress_notes_for(demo_name, idx) {
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
    demo: &'static mlpl_web_demos::Demo,
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
    // Connect mode: route the line through the server (so loaded
    // demos behave like typed lines -- `:models ollama`, `:ask`,
    // and bare expressions hit mlpl-serve instead of the local
    // session). Falls through to local eval when not connected or
    // for non-eligible commands.
    #[cfg(target_arch = "wasm32")]
    if mlpl_web_eval::eval_url::is_connected()
        && dispatch_demo_line_connected(&session, &history, &entries, demo, idx, line)
    {
        return;
    }
    run_demo_line_local(session, history, entries, demo, idx, line);
}

/// Evaluate one demo line in the local in-process WASM session
/// (the non-connect path), in its own `Timeout` tick so the
/// browser paints between lines, then recurse to the next line.
fn run_demo_line_local(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
    mut entries: Vec<HistoryEntry>,
    demo: &'static mlpl_web_demos::Demo,
    idx: usize,
    line: &'static str,
) {
    let session_next = Rc::clone(&session);
    let history_next = history.clone();
    gloo::timers::callback::Timeout::new(0, move || {
        let r = session.borrow().eval_with_values(line);
        let is_error = r.display.starts_with("error:");
        if !is_error {
            let name = line.split('=').next().unwrap_or(line).trim().to_string();
            let info = mlpl_web_viz3d::events::build_shape_info_full(
                name,
                r.shape,
                r.values,
                r.string_list,
                r.viz_node,
            );
            mlpl_web_viz3d::events::emit(&mlpl_web_viz3d::events::Stage3dEvent {
                step_idx: 0,
                label: line.to_string(),
                output: info,
            });
        }
        replace_running_with_result(&mut entries, line, r.display);
        history.set(entries.clone());
        schedule_demo_line(session_next, history_next, entries, demo, idx + 1);
    })
    .forget();
}

/// Connect-mode demo line: dispatch to the server and continue the
/// demo when the async result lands. Returns false (caller does
/// local eval) for lines the server path does not handle.
#[cfg(target_arch = "wasm32")]
fn dispatch_demo_line_connected(
    session: &Rc<RefCell<WasmSession>>,
    history: &UseStateHandle<Vec<HistoryEntry>>,
    entries: &[HistoryEntry],
    demo: &'static mlpl_web_demos::Demo,
    idx: usize,
    line: &'static str,
) -> bool {
    let mut entries_c = entries.to_vec();
    let hist = history.clone();
    let sess = Rc::clone(session);
    let cont: mlpl_web_eval::eval::ResultCb = Box::new(move |display: String| {
        replace_running_with_result(&mut entries_c, line, display);
        hist.set(entries_c.clone());
        schedule_demo_line(sess, hist.clone(), entries_c, demo, idx + 1);
    });
    // Connect-only demos (MLX/CUDA tier) run their heavy lines on the
    // server so the browser main thread stays responsive (no "page
    // unresponsive" freeze) and the GPU does the work. CPU-tier demos
    // route only :ask/:models and keep plain lines local (3D).
    let route_all = mlpl_web_demos::capability_for(demo.name).requires_connect;
    crate::connect::dispatch_demo_line(line, entries, route_all, cont)
}
