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
/// Overridable per page via `?ollama=<url>` / `?model=<name>` so
/// the demo can point at a remote GPU box + a bigger tool-capable
/// model (e.g. `?ollama=http://large12:11434&model=qwen2.5-coder:14b`).
/// mlpl-serve runs the `llm_call` server-side, so the target host
/// just has to be reachable from the server (no browser CORS).
const ASK_URL: &str = "http://localhost:11434";
const ASK_MODEL: &str = "qwen2.5:0.5b";

/// System/meta preamble prepended to every `:ask` so even broad
/// questions ("describe this environment") are grounded in what
/// MLPL is and what context the user can surface, rather than the
/// model guessing about generic OS/web environments.
const ASK_SYSTEM: &str = "You are an assistant embedded INSIDE the sw-MLPL REPL -- an APL/J/BQN-inspired array and tensor language for machine learning, with a 3D visualization playground (the REPL renders each result as a 3D sculpture: tensors as grids/bars, attention as heatmaps, models as Sankey diagrams). You are NOT a generic cloud/AWS/web assistant; EVERY question is about sw-MLPL. The user runs MLPL expressions in this REPL; `:help` lists the builtins and `:history` shows recent commands. The user's recent REPL activity and any selected 3D sculpture are provided below as your context -- use them. Answer concisely and specifically about sw-MLPL.";

/// Read a URL query parameter, falling back to `default` when it
/// is absent or empty. (`name` is a fixed literal, never user
/// input, so the inlined script is safe.)
fn query_param(name: &str, default: &str) -> String {
    let script = format!("new URLSearchParams(location.search).get('{name}') || ''");
    js_sys::eval(&script)
        .ok()
        .and_then(|v| v.as_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Plain-text summary of the sculpture the user is currently
/// inspecting (via the `window.__stage3d_context()` JS hook), or
/// empty when nothing is selected.
fn selection_context() -> String {
    js_sys::eval("window.__stage3d_context ? window.__stage3d_context() : ''")
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default()
}

/// Summarize the recent REPL activity (last few command/result
/// pairs) so `:ask` can answer questions about "what is being run
/// in the REPL" -- an in-context REPL assistant. Newest entries
/// are kept; long outputs are truncated char-safely.
fn repl_history_context(history: &[HistoryEntry]) -> String {
    let mut recent: Vec<String> = history
        .iter()
        .rev()
        .filter(|e| matches!(e.kind, EntryKind::Command))
        .take(6)
        .map(|e| {
            let out: String = e.output.trim().chars().take(180).collect();
            format!("mlpl> {} => {}", e.input.trim(), out)
        })
        .collect();
    recent.reverse();
    recent.join(" | ")
}

/// Build the `:ask` system message: meta preamble + recent REPL
/// activity + the selected sculpture (if any). This goes in
/// Ollama's `system` role (not the prompt), which weak models
/// follow far better. The question is sent as the user prompt.
fn build_ask_system(history: &[HistoryEntry]) -> String {
    let mut p = ASK_SYSTEM.to_string();
    let recent = repl_history_context(history);
    if !recent.is_empty() {
        p.push_str(&format!(" Recent REPL activity (oldest first): {recent}."));
    }
    let sel = selection_context();
    if !sel.is_empty() {
        p.push_str(&format!(" Selected 3D sculpture: {sel}."));
    }
    p
}

/// Map a submitted line to the program to send to the server.
/// `:ask <question>` becomes a 4-arg `llm_call` -- the question is
/// the user prompt and the grounding/context rides in the `system`
/// field. A bare expression passes through; any other slash-command
/// returns `None` to stay local.
fn connect_program(line: &str, history: &[HistoryEntry]) -> Option<String> {
    let t = line.trim_start();
    if let Some(q) = t.strip_prefix(":ask ") {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let question = esc(q.trim().trim_matches('"').trim());
        let system = esc(&build_ask_system(history));
        let (url, model) = ask_endpoint();
        return Some(format!(
            "llm_call(\"{url}\", \"{question}\", \"{model}\", \"{system}\")"
        ));
    }
    if t.starts_with(':') {
        return None;
    }
    Some(line.to_string())
}

/// Route one demo line through the connected server (connect mode):
/// the `:models ollama` listing, the `:ask` shortcut, or a bare
/// expression. Fires `on_result` with the display string when the
/// server responds. Returns true when it took the line; false for a
/// non-eligible `:` command so the demo runner falls back to local
/// eval. Shared by the demo runner so loaded demos behave like
/// typed lines in connect mode.
pub(crate) fn dispatch_demo_line(
    line: &str,
    history: &[HistoryEntry],
    route_all: bool,
    on_result: mlpl_web_eval::eval::ResultCb,
) -> bool {
    let t = line.trim();
    if t == ":models ollama" || t.starts_with(":models ollama ") {
        let host = t
            .strip_prefix(":models ollama ")
            .map(|h| h.trim().to_string());
        let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
            return false;
        };
        mlpl_web_eval::ollama_fetch::fetch_ollama_models(base, host, on_result);
        return true;
    }
    if t.starts_with(":ask ") {
        if let Some(program) = connect_program(line, history) {
            return mlpl_web_eval::eval_wasm::connect_eval(&program, on_result);
        }
    }
    // `route_all` is set for connect-only demos (MLX/CUDA tier):
    // their heavy training runs server-side so the browser main
    // thread stays responsive (and uses the GPU). CPU-tier demos
    // pass `route_all=false`, so plain MLPL lines stay LOCAL and keep
    // emitting 3D viz (the connect path returns only a display
    // string). Full 3D for server-routed lines arrives with the
    // viz-passthrough step.
    if route_all && !t.starts_with(':') {
        return mlpl_web_eval::eval_wasm::connect_eval(line, on_result);
    }
    false
}

/// Resolve the `(host, model)` for `:ask`: an explicit `?ollama=` /
/// `?model=` page override wins; else the server-configured default
/// primed on connect (`GET /v1/ollama/config`); else the built-in
/// constants.
fn ask_endpoint() -> (String, String) {
    let (def_url, def_model) = mlpl_web_eval::ollama_fetch::ollama_default()
        .unwrap_or_else(|| (ASK_URL.to_string(), ASK_MODEL.to_string()));
    (
        query_param("ollama", &def_url),
        query_param("model", &def_model),
    )
}

/// Connect-mode `:models ollama`: fetch the server's Ollama model
/// list (`GET /v1/ollama/tags`) and render it as a history entry,
/// chaining the rest of the queue. Returns true when it took the
/// line (connect mode + the exact command); false to fall through
/// to local handling.
pub(crate) fn try_ollama_models(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    line: &str,
) -> bool {
    let t = line.trim();
    if t != ":models ollama" && !t.starts_with(":models ollama ") {
        return false;
    }
    let host = t
        .strip_prefix(":models ollama ")
        .map(|h| h.trim().to_string());
    let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
        return false;
    };
    let hist_handle = deps.history.clone();
    let deps_c = deps.clone();
    let queue_c = queue.to_vec();
    let line_in = t.to_string();
    let mut hist_c = history.to_vec();
    mlpl_web_eval::ollama_fetch::fetch_ollama_models(
        base,
        host,
        Box::new(move |result: String| {
            let is_error = result.starts_with("error:");
            hist_c.pop();
            hist_c.push(HistoryEntry {
                input: line_in,
                output: result,
                is_error,
                kind: EntryKind::Command,
            });
            hist_handle.set(hist_c.clone());
            crate::submit::process_next_eval(deps_c, hist_c, queue_c, idx + 1);
        }),
    );
    true
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
    let Some(program) = connect_program(line, history) else {
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
