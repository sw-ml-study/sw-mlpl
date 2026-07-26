use yew::prelude::*;

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
        commit_batch(&deps, new_history, new_cmds, eval_queue);
    })
}

/// Commit the classified batch exactly once: skip empty submissions,
/// then either paint the history or start the eval queue.
fn commit_batch(
    deps: &EvalDeps,
    history: Vec<HistoryEntry>,
    cmds: Vec<String>,
    queue: Vec<String>,
) {
    if cmds.is_empty() {
        return;
    }
    deps.cmd_history.set(cmds);
    deps.cmd_index.set(None);
    deps.input_value.set(String::new());
    if queue.is_empty() {
        deps.history.set(history);
        return;
    }
    process_next_eval(deps.clone(), history, queue, 0);
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
    // `:<cmd> --help` is REPL command help (vs bare args = command
    // input). Intercept it here so it works in both connect + local mode
    // and never routes to the server / Ollama.
    if let Some(help) = crate::help::command_help(trimmed) {
        new_history.push(HistoryEntry {
            input: trimmed.to_string(),
            output: help,
            is_error: false,
            kind: EntryKind::Command,
        });
        return;
    }
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
    // Decide telemetry remoteness BEFORE the marker renders, so the live
    // panel only mounts for server-side evals (not browser-local ones).
    #[cfg(target_arch = "wasm32")]
    crate::connect::begin_eval_telemetry(&line, &history);
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
                label: mlpl_eval_core::indent_source(line),
                output: info,
            });
        }
        HistoryEntry {
            input: line.to_string(),
            output: r.display,
            is_error,
            kind: EntryKind::Command,
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        eval_one_line(deps, line)
    }
}

/// Evaluate one REPL line: canned command replies (`:help`,
/// `:history`, `:status`, ...) come from `help::command_reply`;
/// anything else evaluates in the local WASM session.
fn eval_one_line(deps: &EvalDeps, trimmed: &str) -> HistoryEntry {
    if let Some(output) = crate::help::command_reply(deps, trimmed) {
        return HistoryEntry {
            input: trimmed.to_string(),
            output,
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
