//! `:upload <name>` REPL command handler. Saga 29 step 016;
//! extracted to its own file in step 018 to keep
//! `handlers.rs` under the 500-line file budget after the
//! spinner pipeline addition.

use web_sys::HtmlInputElement;

use crate::handlers::EvalDeps;
use crate::state::{EntryKind, HistoryEntry};

/// Parse `:upload <name>` (or `:upload` with no arg). Returns
/// `Some(name)` if the line is the upload command, `None`
/// otherwise. An empty name means the user typed `:upload`
/// without a target.
pub(crate) fn parse_upload_command(line: &str) -> Option<String> {
    let rest = line.strip_prefix(":upload")?;
    Some(rest.trim().to_string())
}

/// Saga 29 step 016: handle one `:upload <name>` line.
/// Returns the HistoryEntry that should appear for this
/// command -- usage / bad-name / not-mounted error message,
/// or the "picker opened" confirmation.
pub(crate) fn handle_upload_command(deps: &EvalDeps, trimmed: &str, name: &str) -> HistoryEntry {
    if name.is_empty() {
        return usage_entry(trimmed);
    }
    if !is_valid_identifier(name) {
        return bad_name_entry(trimmed, name);
    }
    deps.pending_upload_name.set(Some(name.to_string()));
    let Some(input) = deps.upload_input_ref.cast::<HtmlInputElement>() else {
        return HistoryEntry {
            input: trimmed.to_string(),
            output: "error: upload file input not mounted (refresh and retry)".into(),
            is_error: true,
            kind: EntryKind::Command,
        };
    };
    input.set_value("");
    input.click();
    HistoryEntry {
        input: trimmed.to_string(),
        output: format!(
            "Opened the file picker. Pick a photo to bind `{name}` as \
             `Ok({{pixels: [1, 3, 64, 64], h: 64, w: 64}})`, or close \
             the picker to bind `{name} = Err(\"cancelled\")`."
        ),
        is_error: false,
        kind: EntryKind::Command,
    }
}

pub(crate) fn is_valid_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn usage_entry(input: &str) -> HistoryEntry {
    HistoryEntry {
        input: input.to_string(),
        output: "usage: `:upload <name>` -- e.g. `:upload x`. \
                 Picks a photo via the browser file dialog and binds \
                 `<name> = Ok({pixels: [1, 3, 64, 64], h: 64, w: 64})` \
                 on success, `Err(\"cancelled\")` on dismiss."
            .to_string(),
        is_error: true,
        kind: EntryKind::Command,
    }
}

fn bad_name_entry(input: &str, name: &str) -> HistoryEntry {
    HistoryEntry {
        input: input.to_string(),
        output: format!(
            "error: `{name}` is not a valid identifier. Use letters, \
             digits, and underscores; the first character must be a \
             letter or underscore."
        ),
        is_error: true,
        kind: EntryKind::Command,
    }
}
