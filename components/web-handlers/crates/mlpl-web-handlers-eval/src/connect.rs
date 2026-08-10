//! Connect-mode eval dispatch. When the page is opened with
//! `?connect=<url>`, real expressions (and the `:ask` Ollama
//! shortcut) are sent to `mlpl-serve` and evaluated server-side,
//! so `llm_call` / MLX-GPU work run there and the browser never
//! blocks (or panics on WASM-unsupported HTTP/time). Whole module
//! is WASM-only; native builds compile it away.

#![cfg(target_arch = "wasm32")]

use mlpl_web_eval::state::{EntryKind, HistoryEntry};
use mlpl_web_handlers_upload::eval_deps::EvalDeps;

/// Map a submitted line to the program to send to the server.
/// `:ask <question>` becomes a 4-arg `llm_call` -- the question is
/// the user prompt and the grounding/context rides in the `system`
/// field. A bare expression passes through; any other slash-command
/// returns `None` to stay local.
fn connect_program(line: &str, history: &[HistoryEntry]) -> Option<String> {
    let t = line.trim_start();
    if let Some(q) = t.strip_prefix(":ask ") {
        return Some(mlpl_web_ask::prompt::ask_program(q, history));
    }
    if is_inspect_command(t) || is_colon_call_expr(t) {
        return Some(t.to_string());
    }
    if t.starts_with(':') {
        // Colon lines the CLIENT does not handle itself route to the
        // server too: its inspect surface is a superset and it owns
        // the "`:disp` is a builtin REFERENCE" hint (the user's
        // bindings live there in connect mode).
        const CLIENT_LOCAL: &[&str] = &[
            ":help", ":history", ":clear", ":upload", ":2d", ":3d", ":reset", ":ask", ":connect",
            ":status",
        ];
        let word = t.split_whitespace().next().unwrap_or(t);
        if CLIENT_LOCAL.contains(&word) {
            return None;
        }
        return Some(t.to_string());
    }
    Some(line.to_string())
}

/// Workspace-introspection commands ride /eval to the server: in
/// connect mode the session (vars, models, u: fns defined by
/// prefers_connect demos) lives THERE, so :fns / :list must ask it.
/// `:name(...)` is a builtin-reference CALL expression, not a
/// command -- route it to the server like any other program (the
/// user's bindings live there in connect mode).
fn is_colon_call_expr(t: &str) -> bool {
    let Some(rest) = t.strip_prefix(':') else {
        return false;
    };
    let n = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .count();
    n > 0 && rest[n..].starts_with('(')
}

fn is_inspect_command(t: &str) -> bool {
    const INSPECT_CMDS: &[&str] = &[
        ":vars",
        ":fns",
        ":models",
        ":tokenizers",
        ":tags",
        ":untag",
        ":wsid",
        ":experiments",
        ":introspect",
        ":describe",
        ":list",
    ];
    let cmd = t.split_whitespace().next().unwrap_or(t);
    INSPECT_CMDS.contains(&cmd)
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
        // `_auto`: train-bearing lines stream per-step metrics (live
        // loss panel); everything else keeps the JSON path + 3D viz.
        return mlpl_web_eval::eval_wasm::connect_eval_auto(line, on_result);
    }
    false
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
pub(crate) fn chain_entry(
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
        crate::clear::post_reset(deps, history, queue, idx);
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
    // Workspace introspection is SPLIT in connect mode: CPU/browser
    // demos define fns/vars locally (to keep 3D viz) while other work
    // runs server-side. So for an inspect command, capture the LOCAL
    // env's answer now and merge it (labelled) with the server's, so
    // `:fns` shows both a browser-defined `u:life20` and server fns.
    let local_inspect =
        is_inspect_command(line.trim_start()).then(|| deps.session.borrow().eval(line.trim()));
    let hist_handle = deps.history.clone();
    let deps_c = deps.clone();
    let queue_c = queue.to_vec();
    let line_c = line.to_string();
    let mut hist_c = history.to_vec();
    // Capture THIS eval's telemetry generation now (synchronous with its
    // begin()); read only that gen_id's sparkline when the result lands, so
    // a concurrent watch/eval can't intermix.
    let gen_id = mlpl_web_eval::telemetry_trace::current_gen();
    mlpl_web_eval::eval_wasm::connect_eval_auto(
        &program,
        Box::new(move |result: String| {
            let is_error = result.starts_with("error:");
            let output = if let Some(local) = &local_inspect {
                // Merge, each side labelled. (No telemetry for inspect.)
                format!(
                    "[this browser]\n{}\n\n[connected server]\n{}",
                    local.trim(),
                    result.trim()
                )
            } else {
                // Persist the backend-load sparkline collected by the live
                // panel so the trace (incl. a brief GPU blip) survives the
                // marker being replaced. Only when samples were collected.
                let o = match (is_error, mlpl_web_eval::telemetry_trace::summary(gen_id)) {
                    (false, Some(tel)) => format!("{result}\n{tel}"),
                    _ => result,
                };
                // Persist the streamed live-loss curve: the final chart +
                // one-line record under the result once the panel unmounts
                // (chart first, matching the during-training layout).
                match (is_error, mlpl_web_render_live::embed::final_report(gen_id)) {
                    (false, Some(report)) => format!("{o}{report}"),
                    _ => o,
                }
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
