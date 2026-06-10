use yew::prelude::*;

use crate::help::help_text;
use mlpl_web_eval::state::{EntryKind, HistoryEntry};
use mlpl_web_handlers_upload::eval_deps::EvalDeps;
use mlpl_web_handlers_upload::upload_cmd::{handle_upload_command, parse_upload_command};

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
            classify_line(
                &deps,
                line,
                &mut new_history,
                &mut new_cmds,
                &mut eval_queue,
            );
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
        process_next_eval(deps.clone(), new_history, eval_queue, 0);
    })
}

/// Classify one submitted line: `:clear` / `:3d ...` / upload commands
/// run immediately; everything else is queued for the WASM evaluator.
fn classify_line(
    deps: &EvalDeps,
    line: String,
    new_history: &mut Vec<HistoryEntry>,
    new_cmds: &mut Vec<String>,
    eval_queue: &mut Vec<String>,
) {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return;
    }
    new_cmds.push(trimmed.to_string());
    if trimmed == ":clear" {
        deps.session.borrow().clear();
        new_history.clear();
        if *deps.show_3d {
            let _ = js_sys::eval("window.__stage3d_clear && window.__stage3d_clear()");
        }
        return;
    }
    if let Some(cmd) = mlpl_web_viz3d::toggle::parse_3d_command(trimmed) {
        apply_3d_command(deps, cmd);
        return;
    }
    if let Some(name) = parse_upload_command(trimmed) {
        new_history.push(handle_upload_command(deps, trimmed, &name));
        return;
    }
    eval_queue.push(trimmed.to_string());
}

fn apply_3d_command(deps: &EvalDeps, cmd: mlpl_web_viz3d::toggle::Viz3dCmd) {
    match cmd {
        mlpl_web_viz3d::toggle::Viz3dCmd::On => deps.show_3d.set(true),
        mlpl_web_viz3d::toggle::Viz3dCmd::Off => deps.show_3d.set(false),
        mlpl_web_viz3d::toggle::Viz3dCmd::Reset => {
            let _ = js_sys::eval("window.__stage3d_reset_view && window.__stage3d_reset_view()");
        }
    }
}

/// Saga 29 step 018: recursive Timeout-yielding eval loop for
/// REPL submissions. Pushes a Running marker, yields, evals,
/// replaces marker with result, recurses to next line.
pub(crate) fn process_next_eval(
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
    crate::running::push_running_marker(&mut history, &line);
    deps.history.set(history.clone());
    // Connect mode (?connect=<url>): `:connect list` lists the
    // server's Ollama models; real expressions + the `:ask`
    // shortcut route to mlpl-serve (async, server-side) so the
    // browser never blocks. Each returns true when it took the line.
    // A pending `:reset` confirmation (armed on the previous line) wins
    // over every other interpretation of this line -- so the y/N answer
    // is never lexed as MLPL.
    #[cfg(target_arch = "wasm32")]
    if crate::connect::try_reset_answer(&deps, &history, &queue, idx, &line) {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    if crate::connect::try_ollama_models(&deps, &history, &queue, idx, &line) {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    if crate::connect::try_status(&deps, &history, &queue, idx, &line) {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    if crate::connect::try_reset(&deps, &history, &queue, idx, &line) {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    if crate::connect::try_connect_eval(&deps, &history, &queue, idx, &line) {
        return;
    }
    let deps_next = deps.clone();
    let queue_next = queue.clone();
    gloo::timers::callback::Timeout::new(0, move || {
        let entry = eval_one_line_with_3d(&deps_next, &line);
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
        return crate::running::train_caption(stripped);
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

fn eval_one_line_with_3d(deps: &EvalDeps, line: &str) -> HistoryEntry {
    if line == ":help" || line.starts_with(':') {
        return eval_one_line(deps, line);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let r = deps.session.borrow().eval_with_values(line);
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
        return HistoryEntry {
            input: line.to_string(),
            output: r.display,
            is_error,
            kind: EntryKind::Command,
        };
    }
    #[cfg(not(target_arch = "wasm32"))]
    eval_one_line(deps, line)
}

/// `:history` output -- the recent REPL command lines, so the
/// user (and the `:ask` LLM, which also receives this) can see
/// what has been run.
fn history_listing(deps: &EvalDeps) -> String {
    let lines: Vec<String> = deps
        .history
        .iter()
        .filter(|e| matches!(e.kind, EntryKind::Command))
        .map(|e| format!("mlpl> {}", e.input.trim()))
        .collect();
    if lines.is_empty() {
        "(no commands run yet)".to_string()
    } else {
        lines.join("\n")
    }
}

/// Shown for `:ask` when there is no connected server (e.g. the
/// public live demo). `:ask` needs a server to reach an LLM, so we
/// give a clear notice instead of lexing the question as MLPL
/// (which errors on punctuation like `?`).
const ASK_NEEDS_SERVER: &str = "`:ask` is not available on the public demo -- it needs a connected mlpl-serve with Ollama running. Run `mlpl-serve` on a local machine (with `ollama serve`), then open this REPL with `?connect=<server-url>` appended to the page URL (e.g. `...?connect=http://host:6464`). The CUDA / MLX demos additionally need that server on a host with the matching GPU (Linux+NVIDIA for CUDA, Apple Silicon for MLX).";

/// Shown for `:status` when no server is connected. Connect mode
/// answers `:status` with a live backend report (devices + CPU/RAM/
/// GPU/VRAM); here there is no backend to probe.
const STATUS_LOCAL: &str = "Status: 0 backends connected -- local (browser) mode.\n  device  : cpu (browser WASM)\n  Live CPU/GPU/RAM/VRAM telemetry, :ask, and the CUDA/MLX demos need a\n  connected mlpl-serve. Start one (mlpl-serve --bind 0.0.0.0:6464\n  --auth required), then append ?connect=<server-url> to this page's URL\n  (e.g. ?connect=http://host:6464).";

/// Text for `:connect <arg>` outside the async connect path. `set
/// <model>` selects the `:ask` model for the session (local/sync);
/// `list` / bare need a connected server (handled upstream when one is
/// connected, so here it means "not connected").
fn connect_command_text(arg: &str) -> String {
    if let Some(name) = arg.strip_prefix("set ").map(str::trim) {
        if name.is_empty() {
            return "usage: :connect set <model>   (see :connect list)".to_string();
        }
        #[cfg(target_arch = "wasm32")]
        mlpl_web_eval::ollama_fetch::set_selected_model(name);
        format!("ask model set to {name}")
    } else if arg == "list" || arg.is_empty() {
        format!(
            "`:connect list` needs a connected mlpl-serve to query its Ollama models. {ASK_NEEDS_SERVER}"
        )
    } else {
        format!("unknown :connect subcommand '{arg}' (try: list  |  set <model>)")
    }
}

fn eval_one_line(deps: &EvalDeps, trimmed: &str) -> HistoryEntry {
    if trimmed == ":ask" || trimmed.starts_with(":ask ") {
        return HistoryEntry {
            input: trimmed.to_string(),
            output: ASK_NEEDS_SERVER.to_string(),
            is_error: false,
            kind: EntryKind::Command,
        };
    }
    if trimmed == ":connect" || trimmed.starts_with(":connect ") {
        // `:connect set <model>` is local/sync; `:connect list` needs a
        // connected server (the async listing is handled upstream in
        // connect mode, so reaching here means not connected).
        return HistoryEntry {
            input: trimmed.to_string(),
            output: connect_command_text(trimmed.strip_prefix(":connect").unwrap_or("").trim()),
            is_error: false,
            kind: EntryKind::Command,
        };
    }
    if trimmed == ":help" {
        return HistoryEntry {
            input: trimmed.to_string(),
            output: help_text(),
            is_error: false,
            kind: EntryKind::Command,
        };
    }
    if trimmed == ":history" {
        return HistoryEntry {
            input: trimmed.to_string(),
            output: history_listing(deps),
            is_error: false,
            kind: EntryKind::Command,
        };
    }
    if trimmed == ":status" {
        // Connect-mode `:status` is handled async upstream
        // (connect::try_status); reaching here means no server is
        // connected, so report local browser mode.
        return HistoryEntry {
            input: trimmed.to_string(),
            output: STATUS_LOCAL.to_string(),
            is_error: false,
            kind: EntryKind::Command,
        };
    }
    if trimmed == ":reset" {
        // Connect-mode `:reset` is handled async upstream
        // (connect::try_reset); reaching here means no server, so
        // there is no backend work to cancel.
        return HistoryEntry {
            input: trimmed.to_string(),
            output: "Nothing to reset -- local browser mode, no connected server.".to_string(),
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
