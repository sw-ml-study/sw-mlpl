//! Connect-mode eval dispatch. When the page is opened with
//! `?connect=<url>`, real expressions (and the `:ask` Ollama
//! shortcut) are sent to `mlpl-serve` and evaluated server-side,
//! so `llm_call` / MLX-GPU work run there and the browser never
//! blocks (or panics on WASM-unsupported HTTP/time). Whole module
//! is WASM-only; native builds compile it away.

#![cfg(target_arch = "wasm32")]

use mlpl_web_eval::state::{EntryKind, HistoryEntry};
use mlpl_web_handlers_upload::eval_deps::EvalDeps;

/// Default Ollama endpoint + model backing the `:ask` shortcut.
const ASK_URL: &str = "http://localhost:11434";
const ASK_MODEL: &str = "qwen2.5:0.5b";

/// Map a submitted line to the program to send to the server.
/// `:ask <question>` becomes an `llm_call`; a bare expression
/// passes through unchanged; any other slash-command returns
/// `None` to stay local (UI state lives in the browser).
fn connect_program(line: &str) -> Option<String> {
    let t = line.trim_start();
    if let Some(q) = t.strip_prefix(":ask ") {
        let esc = q.trim().replace('\\', "\\\\").replace('"', "\\\"");
        return Some(format!(
            "llm_call(\"{ASK_URL}\", \"{esc}\", \"{ASK_MODEL}\")"
        ));
    }
    if t.starts_with(':') {
        return None;
    }
    Some(line.to_string())
}

/// Dispatch `line` to the connected server (async) when a connect
/// URL is set and the line is server-eligible, chaining the rest
/// of the queue in the result callback so line order is kept.
/// Returns true when it took the eval; false to let the caller
/// evaluate locally.
pub(crate) fn try_connect_eval(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    line: &str,
) -> bool {
    let Some(program) = connect_program(line) else {
        return false;
    };
    let hist_handle = deps.history.clone();
    let deps_c = deps.clone();
    let queue_c = queue.to_vec();
    let line_c = line.to_string();
    let mut hist_c = history.to_vec();
    mlpl_web_eval::eval_wasm::connect_eval(
        &program,
        Box::new(move |result: String| {
            let is_error = result.starts_with("error:");
            hist_c.pop();
            hist_c.push(HistoryEntry {
                input: line_c.clone(),
                output: result,
                is_error,
                kind: EntryKind::Command,
            });
            hist_handle.set(hist_c.clone());
            crate::submit::process_next_eval(deps_c, hist_c, queue_c, idx + 1);
        }),
    )
}
