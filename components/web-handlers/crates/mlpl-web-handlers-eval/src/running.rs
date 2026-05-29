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
