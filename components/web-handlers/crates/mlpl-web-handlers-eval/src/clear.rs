//! The REPL "Clear" button callback: drops history, asks the
//! shared WasmSession to forget every binding, and fires the
//! Three.js stage reset hook. Extracted from the original
//! handlers.rs facade during saga 82.

use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use mlpl_web_eval::state::{EntryKind, HistoryEntry};
use mlpl_web_handlers_upload::eval_deps::EvalDeps;
use yew::prelude::*;

pub fn make_clear(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        session.borrow().clear();
        history.set(Vec::new());
        let _ = js_sys::eval("window.__stage3d_clear && window.__stage3d_clear()");
    })
}

// Connect-mode `:reset` POST (moved from connect.rs, which sat at
// its file-LOC budget): session-clearing is this module's job, and
// the server-side reset is the connect-mode half of that story.
/// POST `/v1/reset` (confirmed): cancel all in-flight evals on the
/// server, render the cancel count, and chain the queue.
pub(crate) fn post_reset(deps: &EvalDeps, history: &[HistoryEntry], queue: &[String], idx: usize) {
    let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
        crate::connect::chain_entry(
            deps,
            history,
            queue,
            idx,
            (":reset", "error: not connected", true),
        );
        return;
    };
    let hist_handle = deps.history.clone();
    let deps_c = deps.clone();
    let queue_c = queue.to_vec();
    let mut hist_c = history.to_vec();
    mlpl_web_eval::stats_fetch::fetch_reset(
        base,
        Box::new(move |result: String| {
            let is_error = result.starts_with("error:");
            hist_c.pop();
            hist_c.push(HistoryEntry {
                input: ":reset".to_string(),
                output: result,
                is_error,
                kind: EntryKind::Command,
            });
            hist_handle.set(hist_c.clone());
            crate::submit::process_next_eval(deps_c, hist_c, queue_c, idx + 1);
        }),
    );
}
