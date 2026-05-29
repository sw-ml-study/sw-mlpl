//! Tab-completion engine: builds the candidate pool from
//! static REPL commands + keywords + caller-supplied builtin
//! names, then decides whether to apply, popup, or do nothing.

use crate::prefix::{apply_completion, extract_prefix, match_candidates};

/// Static REPL command names recognized by mlpl-eval's
/// inspect dispatcher (mirrors `inspect.rs::topic_output`
/// plus the arg-taking arms). Stays in sync manually.
pub const REPL_COMMANDS: &[&str] = &[
    ":builtins",
    ":describe",
    ":experiments",
    ":fns",
    ":help",
    ":introspect",
    ":models",
    ":tags",
    ":untag",
    ":upload",
    ":vars",
    ":version",
    ":wsid",
];

/// Static MLPL language keywords. Small enough to enumerate;
/// stays in sync with parser literals.
pub const KEYWORDS: &[&str] = &["train", "repeat", "experiment", "for", "in", "param"];

/// The completion-popup trigger predicate. `ctrl_key` is
/// `KeyboardEvent::ctrl_key()`; `code` is
/// `KeyboardEvent::code()` (returns the physical key name,
/// e.g. `"Space"`, layout-independent). Returns true for
/// Ctrl+Space, which fires the completion lookup. Pure +
/// trivial; lives here so it's unit-testable.
#[must_use]
pub fn is_completion_trigger(ctrl_key: bool, code: &str) -> bool {
    ctrl_key && code == "Space"
}

/// One Tab-press outcome.
pub enum TabMatch {
    /// No matches. Input stays as-is.
    None,
    /// Exactly one match. Apply it and clear any popup.
    Apply { input: String, cursor: usize },
    /// Multiple matches. Show the popup.
    Popup(Vec<String>),
}

fn build_all_candidates<'a, I>(builtins: I) -> Vec<&'a str>
where
    I: Iterator<Item = &'a str>,
{
    let mut all: Vec<&str> = REPL_COMMANDS
        .iter()
        .chain(KEYWORDS.iter())
        .copied()
        .chain(builtins)
        .collect();
    all.sort();
    all.dedup();
    all
}

/// Resolve a completion keypress: build the candidate list
/// from static names + runtime builtins, match the prefix at
/// `cursor`, decide how to react.
pub fn compute_tab_match<'a, I>(input: &str, cursor: usize, builtins: I) -> TabMatch
where
    I: Iterator<Item = &'a str>,
{
    let (_, prefix) = extract_prefix(input, cursor);
    if prefix.is_empty() {
        return TabMatch::None;
    }
    let all = build_all_candidates(builtins);
    let candidates = match_candidates(prefix, &all);
    match candidates.len() {
        0 => TabMatch::None,
        1 => {
            let (input, cursor) = apply_completion(input, cursor, &candidates[0]);
            TabMatch::Apply { input, cursor }
        }
        _ => TabMatch::Popup(candidates),
    }
}
