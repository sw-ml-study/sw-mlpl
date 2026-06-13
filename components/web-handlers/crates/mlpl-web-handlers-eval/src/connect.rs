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
const ASK_SYSTEM: &str = "You are an assistant embedded INSIDE the sw-MLPL REPL -- an APL/J/BQN-inspired array and tensor language for machine learning, with a 3D visualization playground (the REPL renders each result as a 3D sculpture: tensors as grids/bars, attention as heatmaps, models as Sankey diagrams). You are NOT a generic cloud/AWS/web assistant; EVERY question is about sw-MLPL. \
CRITICAL: Do NOT invent, guess, or hallucinate commands, syntax, model names, or features. If you are unsure, say so plainly and tell the user to run `:help`. The ONLY REPL commands are exactly these -- never claim any other exists: :help, :help <topic>, :<cmd> --help, :vars, :models, :tokenizers, :fns, :builtins, :describe <name>, :history, :experiments, :wsid, :status, :status watch, :reset, :ask <prompt>, :connect list, :connect set <model>, :3d, :2d, :clear, :upload <name>. There is NO :load_model, NO :model, NO :search. \
DO NOT WRITE MLPL CODE unless you are CERTAIN of the exact syntax (ideally only code you can see in this session's context): MLPL is an array language, NOT Python -- it has NO `+=`/`-=`, NO `==` (use `eq(a, b)`), NO lambdas or `->`, NO `filter`/`map`/`print`/`length`/`strcat`/`append`, and `device(\"...\")`, `experiment`, and `train`/`repeat` ALWAYS take a `{ ... }` block (a bare `device(\"mlx\")` is a syntax error). If you are not sure code will run, DESCRIBE the approach in plain words and tell the user to read the demo/its literate walkthrough and `:help` -- never emit a code block you have not actually seen run. When the user asks how to do something in the REPL: for command help tell them to run `:help` (lists all commands) or `:<cmd> --help` (help for one command, e.g. `:ask --help`); to see or change which LLM answers their `:ask`, tell them `:connect list` (lists the installed Ollama models) then `:connect set <name>`. \
The user's recent REPL activity and any selected 3D sculpture are provided below as your context -- use them. Answer concisely and specifically about sw-MLPL.";

/// Compact MLPL syntax cheat-sheet prepended to the builtin signatures in
/// [`mlpl_reference`]. The CORRECT forms (vs the Python-isms the prompt
/// forbids), so when code is warranted the model writes valid MLPL.
const MLPL_SYNTAX: &str = " MLPL quick reference -- use EXACTLY these forms. \
Assign `name = expr`; comment `# ...`; compare with `eq(a,b)` / `gt(a,b)` / `lt(a,b)` (there is no `==`/`<`/`>`); \
EVERY block uses braces: `device(\"mlx\") { ... }`, `experiment \"name\" { ... }`, `train N { ... }`, `repeat N { ... }`, `if c { ... } else { ... }`; \
define a function with `def u:name(a, b) { body }` -- the `u:` prefix is REQUIRED and there is NO `return` (the block's last expression is its value) -- then call it `u:name(args)`; iterate with `for x in iota(n) { ... }` or `while cond { ... }`; index/slice a tensor with `take(x, axis, i)`. \
Statements inside a block are separated by `;`. The COMPLETE builtin set follows (call ONLY these exact signatures -- there is no filter/map/print/length/strcat/append):";

/// The compact MLPL reference for the `:ask` system prompt: the syntax
/// cheat-sheet plus every builtin's signature, grouped, sourced from the
/// curated `BUILTIN_GROUPS` table so it never drifts from the real set.
fn mlpl_reference() -> String {
    let mut r = MLPL_SYNTAX.to_string();
    for (group, entries) in mlpl_eval_core::inspect_groups::BUILTIN_GROUPS {
        let sigs: Vec<&str> = entries.iter().map(|&(_, sig, _)| sig).collect();
        r.push_str(&format!(" [{group}] {}.", sigs.join(", ")));
    }
    r
}

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
    // Compact MLPL reference (syntax + builtin signatures) -- real forms.
    p.push_str(&mlpl_reference());
    // Ground the model in the active demo (its "About this demo -- <name>"
    // narration names the task, e.g. tic-tac-toe) so it doesn't guess Othello.
    if let Some(intro) = history
        .iter()
        .rev()
        .find(|e| matches!(e.kind, EntryKind::Narration) && e.input.starts_with("About this demo"))
    {
        let body: String = intro.output.trim().chars().take(400).collect();
        p.push_str(&format!(" Active demo -- {}: {body}.", intro.input.trim()));
    }
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

/// Set up the telemetry trace for the eval about to run. It is `remote`
/// (live panel shown) only when connected AND the line is a server
/// -routed program (a bare expression or `:ask`); browser-local evals
/// `begin(false)` so the panel stays hidden for a computation that runs
/// in the browser, not on the server.
pub(crate) fn begin_eval_telemetry(line: &str, history: &[HistoryEntry]) {
    let remote = mlpl_web_eval::eval::current_connect_url_from_window().is_some()
        && connect_program(line, history).is_some();
    mlpl_web_eval::telemetry_trace::begin(remote);
}

/// Route one demo line through the connected server (connect mode):
/// the `:connect list` listing, the `:ask` shortcut, or a bare
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
    // Demo lines carry trailing `# comment` annotations. Strip them only
    // for matching `:status` / `:connect list` (so `:connect list # ...`
    // is not read as a host and `:status # ...` matches). `:ask` and plain
    // MLPL lines keep the ORIGINAL `line`: `:ask` sends its prompt to
    // Ollama verbatim (a `#` is part of the question, not a comment), and
    // a plain MLPL line may carry a `#` inside a string literal.
    let cmd_line = line.split('#').next().unwrap_or(line);
    let t = cmd_line.trim();
    if t == ":status" {
        let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
            return false;
        };
        let ready = mlpl_web_eval::ollama_fetch::ollama_default().is_some();
        mlpl_web_eval::stats_fetch::fetch_status(base, ready, on_result);
        return true;
    }
    if t == ":connect list" || t.starts_with(":connect list ") {
        let host = t
            .strip_prefix(":connect list ")
            .map(|h| h.trim().to_string());
        let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
            return false;
        };
        mlpl_web_eval::ollama_fetch::fetch_ollama_models(base, host, on_result);
        return true;
    }
    if t.starts_with(":ask ") {
        // Verbatim: the whole prompt (including any `#`) goes to Ollama.
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

/// Resolve the `(host, model)` for `:ask`. Host: `?ollama=` page
/// override, else the connect-primed default, else the constant. Model:
/// a `:connect set <model>` session pick wins, then `?model=`, then the
/// connect-primed default, then the constant.
fn ask_endpoint() -> (String, String) {
    let (def_url, def_model) = mlpl_web_eval::ollama_fetch::ollama_default()
        .unwrap_or_else(|| (ASK_URL.to_string(), ASK_MODEL.to_string()));
    let model = mlpl_web_eval::ollama_fetch::selected_model()
        .unwrap_or_else(|| query_param("model", &def_model));
    (query_param("ollama", &def_url), model)
}

/// Connect-mode `:connect list` (and bare `:connect`, which aliases to
/// it when connected): fetch the server's Ollama model list (`GET
/// /v1/ollama/tags`) and render it as a history entry, chaining the rest
/// of the queue. Returns true when it took the line (connect mode + the
/// command); false to fall through to local handling -- so when NOT
/// connected, bare `:connect` still reaches the "needs a server" message.
pub(crate) fn try_ollama_models(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    line: &str,
) -> bool {
    let t = line.trim();
    if t != ":connect" && t != ":connect list" && !t.starts_with(":connect list ") {
        return false;
    }
    let host = t
        .strip_prefix(":connect list ")
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

/// Connect-mode `:status`: probe the server's `/v1/devices` +
/// `/v1/stats` and render a self-test report (devices, live
/// CPU/RAM/GPU/VRAM, Ollama state) as a history entry, chaining the
/// rest of the queue. Returns true when it took the line (connect mode
/// + the exact command); false to fall through to the local handler.
pub(crate) fn try_status(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    line: &str,
) -> bool {
    let t = line.trim();
    if t != ":status" && t != ":status watch" {
        return false;
    }
    let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
        return false;
    };
    let hist_handle = deps.history.clone();
    let deps_c = deps.clone();
    let queue_c = queue.to_vec();
    let input = t.to_string();
    let mut hist_c = history.to_vec();
    let cb: mlpl_web_eval::eval::ResultCb = Box::new(move |result: String| {
        let is_error = result.starts_with("error:");
        hist_c.pop();
        hist_c.push(HistoryEntry {
            input: input.clone(),
            output: result,
            is_error,
            kind: EntryKind::Command,
        });
        hist_handle.set(hist_c.clone());
        crate::submit::process_next_eval(deps_c, hist_c, queue_c, idx + 1);
    });
    if t == ":status watch" {
        // bounded burst (~12 samples, ~3.6s) -> persisted sparkline
        mlpl_web_eval::telemetry_trace::watch(base, 12, cb);
    } else {
        let ollama_ready = mlpl_web_eval::ollama_fetch::ollama_default().is_some();
        mlpl_web_eval::stats_fetch::fetch_status(base, ollama_ready, cb);
    }
    true
}

/// Synchronously replace the trailing running-marker with a finished
/// command entry and chain the rest of the queue. For connect commands
/// that resolve without a fetch (the `:reset` prompt and its abort).
fn chain_entry(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    entry: (&str, &str, bool),
) {
    let (input, output, is_error) = entry;
    let mut hist_c = history.to_vec();
    hist_c.pop();
    hist_c.push(HistoryEntry {
        input: input.to_string(),
        output: output.to_string(),
        is_error,
        kind: EntryKind::Command,
    });
    deps.history.set(hist_c.clone());
    crate::submit::process_next_eval(deps.clone(), hist_c, queue.to_vec(), idx + 1);
}

/// Connect-mode `:reset` (step 1 of 2): ARM the confirmation and print a
/// (y/N) prompt. `:reset` cancels ALL in-flight work on the backend, so
/// it never fires on a stray keystroke -- the POST happens only if the
/// next line confirms (`try_reset_answer`). Returns false in local mode
/// so the "nothing to reset" handler runs instead.
pub(crate) fn try_reset(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    line: &str,
) -> bool {
    if line.trim() != ":reset" || mlpl_web_eval::eval::current_connect_url_from_window().is_none() {
        return false;
    }
    mlpl_web_eval::stats_fetch::arm_reset();
    let prompt = "Cancel ALL in-flight work on the connected backend? This aborts every \
                  running eval / training loop. Type `y` (or `yes`) to confirm; \
                  anything else aborts.  (y/N)";
    chain_entry(deps, history, queue, idx, (":reset", prompt, false));
    true
}

/// Consume a pending `:reset` confirmation. Runs FIRST in the dispatch
/// chain but only acts when `:reset` was armed on the previous line:
/// `y`/`yes` -> POST /v1/reset; anything else -> abort, no changes.
/// Returns true when it consumed the line.
pub(crate) fn try_reset_answer(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    line: &str,
) -> bool {
    if !mlpl_web_eval::stats_fetch::take_reset_armed() {
        return false;
    }
    let ans = line.trim().to_ascii_lowercase();
    if ans == "y" || ans == "yes" {
        post_reset(deps, history, queue, idx);
    } else {
        chain_entry(
            deps,
            history,
            queue,
            idx,
            (line, "reset aborted -- no changes.", false),
        );
    }
    true
}

/// POST `/v1/reset` (confirmed): cancel all in-flight evals on the
/// server, render the cancel count, and chain the queue.
fn post_reset(deps: &EvalDeps, history: &[HistoryEntry], queue: &[String], idx: usize) {
    let Some(base) = mlpl_web_eval::eval::current_connect_url_from_window() else {
        chain_entry(
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
    // Capture THIS eval's telemetry generation now (synchronous with its
    // begin()); read only that gen_id's sparkline when the result lands, so
    // a concurrent watch/eval can't intermix.
    let gen_id = mlpl_web_eval::telemetry_trace::current_gen();
    mlpl_web_eval::eval_wasm::connect_eval(
        &program,
        Box::new(move |result: String| {
            let is_error = result.starts_with("error:");
            // Persist the backend-load sparkline collected by the live
            // panel so the trace (incl. a brief GPU blip) survives the
            // marker being replaced. Only when samples were collected.
            let output = match (is_error, mlpl_web_eval::telemetry_trace::summary(gen_id)) {
                (false, Some(tel)) => format!("{result}\n{tel}"),
                _ => result,
            };
            hist_c.pop();
            hist_c.push(HistoryEntry {
                input: line_c.clone(),
                output,
                is_error,
                kind: EntryKind::Command,
            });
            hist_handle.set(hist_c.clone());
            crate::submit::process_next_eval(deps_c, hist_c, queue_c, idx + 1);
        }),
    )
}
