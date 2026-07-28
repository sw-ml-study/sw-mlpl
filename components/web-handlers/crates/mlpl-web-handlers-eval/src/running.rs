//! Running-marker helpers extracted from handlers_submit.rs to
//! keep that module under the sw-checklist function-count budget.
//! Pure formatting + history-mutation helpers; no callbacks.

use mlpl_web_eval::state::{EntryKind, HistoryEntry};

pub(crate) fn train_caption(stripped: &str) -> &'static str {
    // Parse the iteration count out of `train N {...}`. Small
    // chunks (<=10) get a chunk-shaped message ("the page will
    // un-freeze between chunks"); larger blocks keep the
    // long-duration warning.
    let after_train = stripped["train ".len()..].trim_start();
    let count: usize = after_train
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap_or("")
        .parse()
        .unwrap_or(0);
    if count > 0 && count <= 10 {
        "training chunk... (a few seconds; the page un-freezes between chunks)"
    } else {
        "training... (this can take 30-90 seconds; the page is unresponsive while WASM runs)"
    }
}

/// Saga 29 step 018: push a Running placeholder so the browser
/// paints a spinner before the (blocking) WASM eval starts.
pub(crate) fn push_running_marker(entries: &mut Vec<HistoryEntry>, line: &str) {
    entries.push(HistoryEntry {
        input: line.to_string(),
        output: crate::submit::running_message(line).to_string(),
        is_error: false,
        kind: EntryKind::Running,
    });
}

/// Saga 29 step 018: pop the trailing Running marker and push
/// the actual Command result. Called from inside the
/// post-eval Timeout closure once the WASM session returns.
pub(crate) fn replace_running_with_result(
    entries: &mut Vec<HistoryEntry>,
    line: &str,
    result: String,
) {
    let is_error = result.starts_with("error:");
    entries.pop();
    entries.push(HistoryEntry {
        input: line.to_string(),
        output: result,
        is_error,
        kind: EntryKind::Command,
    });
}

#[cfg(target_arch = "wasm32")]
use mlpl_web_handlers_upload::eval_deps::EvalDeps;

/// Take connect-only commands (`:connect`, `:ask`, `:status`) when the
/// page is NOT connected, replacing the running marker with a friendly
/// note instead of letting the line leak into the MLPL parser (where
/// `:connect list` used to die as "undefined variable: list" on the
/// public demo).
#[cfg(target_arch = "wasm32")]
pub(crate) fn try_connect_only_note(
    deps: &EvalDeps,
    history: &[HistoryEntry],
    queue: &[String],
    idx: usize,
    line: &str,
) -> bool {
    let t = line.trim();
    let Some((note, is_error)) = crate::help::connect_only_note(t) else {
        return false;
    };
    crate::connect::chain_entry(deps, history, queue, idx, (t, &note, is_error));
    true
}

/// Append any progress-note narration panels registered for
/// `(demo_name, line idx)`. Returns whether any were added.
pub(crate) fn push_progress_notes(
    entries: &mut Vec<HistoryEntry>,
    demo_name: &str,
    idx: usize,
) -> bool {
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
