use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use yew::prelude::*;

use crate::running::{push_running_marker, replace_running_with_result};
use mlpl_web_demos::DEMOS;
use mlpl_web_eval::state::{EntryKind, HistoryEntry};

/// Post-demo epilogue appended under the takeaway panel. A demo
/// runs IN the live REPL (bindings persist), so it points the new
/// user at the introspection + experiment commands to keep going.
const NEXT_STEPS: &str = "Next steps? This ran in the live REPL -- your bindings persist, so build on them:\n  \
     :vars       -- variables this demo defined\n  \
     :fns        -- user-defined (u:) functions in play\n  \
     :list       -- everything in the workspace at once\n  \
     :help       -- language + builtin overview (`:help <name>` for one)\n  \
     :describe <name>  -- inspect any variable, model, or builtin\n  \
     experiment \"trial\" { ... }  -- capture _metric scalars to compare runs";

/// Run a demo IN the live REPL: the session's bindings and the
/// transcript both survive, so after several demos `:ask` (grounded
/// in the transcript) can answer about all of them and later demos
/// can inspect earlier results. Only the manual "Reset REPL" button
/// (`make_clear`) starts fresh. The 3D stage still resets per demo
/// so sculptures from unrelated demos don't pile up.
///
/// The intro narration panel frames what the demo does; demo lines
/// follow, and the takeaway narration lands after the final line in
/// `schedule_demo_line`. A `_demo` string is bound in the session so
/// `:describe _demo` reprints the intro + takeaway.
///
/// Lines evaluate asynchronously -- one `Timeout::new(0, ...)` tick
/// each -- so the browser paints between lines (a single heavy line
/// still blocks; see docs/worker-threads.md). The accumulated
/// `entries` vec threads through the recursion by move because a
/// `UseStateHandle` read inside a deferred Timeout sees the stale
/// snapshot from closure-creation time; `history.set` is purely for
/// the UI paint.
pub fn make_run_demo(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
) -> Callback<usize> {
    Callback::from(move |idx: usize| {
        let Some(demo) = DEMOS.get(idx) else {
            return;
        };
        let _ = js_sys::eval("window.__stage3d_clear && window.__stage3d_clear()");
        bind_demo_metadata(&session.borrow(), demo);
        let mut entries = (*history).clone();
        entries.push(HistoryEntry {
            input: format!("About this demo -- {}", demo.name),
            output: intro_with_literate_link(demo),
            is_error: false,
            kind: EntryKind::Narration,
        });
        history.set(entries.clone());
        schedule_demo_line(session.clone(), history.clone(), entries, demo, 0);
    })
}

/// The demo intro, with a "literate walkthrough" link appended when
/// the demo has a companion org-mode HTML (bundled under `literate/`).
/// For connect-only demos the link is the way to see what the demo
/// does without a server -- the page is a real recorded run.
fn intro_with_literate_link(demo: &mlpl_web_demos::Demo) -> String {
    match mlpl_web_demos::literate_for(demo.name) {
        Some(file) => format!(
            "{}\n\n[Emacs Org-mode literate walkthrough](literate/{file}) -- every step run and explained, with the before/after chart.",
            demo.intro
        ),
        None => demo.intro.to_string(),
    }
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

/// Telemetry panel only for SERVER-side lines (browser free to sample):
/// MLX/CUDA compute lines, or any `:ask` (LLM runs server-side). Local
/// evals block the sampler -> no panel (see docs/worker-threads.md).
#[cfg(target_arch = "wasm32")]
fn begin_demo_line_telemetry(demo: &'static mlpl_web_demos::Demo, line: &str) {
    let lt = line.trim_start();
    let cap = mlpl_web_demos::capability_for(demo.name);
    let server_side = mlpl_web_eval::eval_url::is_connected()
        && (lt.starts_with(":ask")
            || ((cap.requires_connect || cap.prefers_connect) && !lt.starts_with(':')));
    mlpl_web_eval::telemetry_trace::begin(server_side);
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
        // Last line ran -- append the takeaway narration panel (with
        // the Next-Steps epilogue) and paint the final history state.
        entries.push(HistoryEntry {
            input: "What just happened".to_string(),
            output: format!("{}\n\n{NEXT_STEPS}", demo.takeaway),
            is_error: false,
            kind: EntryKind::Narration,
        });
        history.set(entries);
        return;
    }
    crate::running::push_progress_notes(&mut entries, demo.name, idx);
    let line = lines[idx];
    #[cfg(target_arch = "wasm32")]
    begin_demo_line_telemetry(demo, line);
    push_running_marker(&mut entries, line);
    history.set(entries.clone());
    // Unconnected connect-only lines (`:connect list`, `:ask`,
    // `:status` in the introspection demo on the PUBLIC page) get
    // their friendly note instead of leaking into the parser.
    #[cfg(target_arch = "wasm32")]
    if let Some((note, _)) = crate::help::connect_only_note(line.trim()) {
        replace_running_with_result(&mut entries, line, note);
        history.set(entries.clone());
        schedule_demo_line(session, history, entries, demo, idx + 1);
        return;
    }
    // Connect mode: route the line through the server (so loaded
    // demos behave like typed lines -- `:connect list`, `:ask`,
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
                label: mlpl_eval_core::indent_source(line),
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
    // This demo line's telemetry generation (begin()'d before its marker).
    let gen_id = mlpl_web_eval::telemetry_trace::current_gen();
    let cont: mlpl_web_eval::eval::ResultCb = Box::new(move |display: String| {
        // Persist the backend-load sparkline below a server-run demo line
        // so a brief GPU blip survives the marker being replaced.
        let out = match (
            display.starts_with("error:"),
            mlpl_web_eval::telemetry_trace::summary(gen_id),
        ) {
            (false, Some(tel)) => format!("{display}\n{tel}"),
            _ => display,
        };
        // Streamed live-loss record (train-bearing connect lines): the
        // final chart + one-line record, chart first like the live view.
        let out = match (
            out.starts_with("error:"),
            mlpl_web_render_live::embed::final_report(gen_id),
        ) {
            (false, Some(report)) => format!("{out}{report}"),
            _ => out,
        };
        replace_running_with_result(&mut entries_c, line, out);
        hist.set(entries_c.clone());
        schedule_demo_line(sess, hist.clone(), entries_c, demo, idx + 1);
    });
    // Connect-only demos (MLX/CUDA tier) run their heavy lines on the
    // server so the browser main thread stays responsive (no "page
    // unresponsive" freeze) and the GPU does the work. CPU-tier demos
    // route only :ask/:models and keep plain lines local (3D).
    let cap = mlpl_web_demos::capability_for(demo.name);
    let route_all = cap.requires_connect || cap.prefers_connect;
    crate::connect::dispatch_demo_line(line, entries, route_all, cont)
}
