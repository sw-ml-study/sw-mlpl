//! Saga 33 step 043 + 045: REPL completion popup. Pure
//! helpers -- prefix extraction, candidate matching,
//! completion application, the trigger-key predicate --
//! plus the static candidate list (REPL commands + language
//! keywords). Builtin names are sourced from
//! `mlpl_runtime::runtime_builtin_names()` at call sites
//! since they live in a different crate.
//!
//! Trigger: **Ctrl+Space** (IDE standard -- VS Code,
//! IntelliJ, Emacs). Tab was the original step-043 trigger
//! but the browser's focus-traversal default beat
//! `preventDefault`; Tab is now reserved for browser
//! element navigation and Ctrl+Space takes over.
//!
//! Tests live in this same file: the helpers are pure and
//! need no Yew runtime.

/// Characters that count as part of a completable token.
/// Alphanumeric + underscore + leading colon (for `:cmds`).
fn is_token_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == ':'
}

/// Find the token prefix ending at `cursor` in `input`.
/// Returns `(prefix_start, prefix)` -- empty `prefix` means
/// the cursor is on whitespace and there's nothing to
/// complete. `cursor` is clamped to `input.len()`.
pub fn extract_prefix(input: &str, cursor: usize) -> (usize, &str) {
    let cursor = cursor.min(input.len());
    // Walk backward from cursor while chars are token-y.
    let mut start = cursor;
    for (i, c) in input[..cursor].char_indices().rev() {
        if is_token_char(c) {
            start = i;
        } else {
            break;
        }
    }
    (start, &input[start..cursor])
}

/// Filter `candidates` to those starting with `prefix`.
/// Returns an owned sorted-deduped Vec to keep callers
/// simple. Empty prefix returns empty (nothing to complete).
pub fn match_candidates(prefix: &str, candidates: &[&str]) -> Vec<String> {
    if prefix.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = candidates
        .iter()
        .filter(|c| c.starts_with(prefix) && **c != prefix)
        .map(|c| (*c).to_string())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Apply `completion` at `cursor` in `input`, replacing the
/// token prefix at the cursor. Returns `(new_input,
/// new_cursor)` where the cursor lands at the end of the
/// inserted completion.
pub fn apply_completion(input: &str, cursor: usize, completion: &str) -> (String, usize) {
    let (start, _prefix) = extract_prefix(input, cursor);
    let cursor = cursor.min(input.len());
    let mut out = String::with_capacity(input.len() + completion.len());
    out.push_str(&input[..start]);
    out.push_str(completion);
    out.push_str(&input[cursor..]);
    let new_cursor = start + completion.len();
    (out, new_cursor)
}

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

/// Saga 33 step 045/046: the completion-popup trigger
/// predicate. `ctrl_key` is `KeyboardEvent::ctrl_key()`;
/// `code` is `KeyboardEvent::code()` (returns the physical
/// key name, e.g. `"Space"`, layout-independent). Returns
/// true for Ctrl+Space, which fires the completion lookup.
/// Pure + trivial; lives here so it's unit-testable.
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

/// Resolve a Tab keypress: build the candidate list from
/// static names + the runtime builtins iterator, match the
/// prefix at `cursor`, decide how to react.
pub fn compute_tab_match<'a, I>(input: &str, cursor: usize, builtins: I) -> TabMatch
where
    I: Iterator<Item = &'a str>,
{
    let (_, prefix) = extract_prefix(input, cursor);
    if prefix.is_empty() {
        return TabMatch::None;
    }
    let mut all: Vec<&str> = REPL_COMMANDS
        .iter()
        .chain(KEYWORDS.iter())
        .copied()
        .chain(builtins)
        .collect();
    all.sort();
    all.dedup();
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

#[cfg(test)]
mod tests {
    use super::{REPL_COMMANDS, apply_completion, extract_prefix, match_candidates};

    #[test]
    fn extract_prefix_on_empty_input_returns_empty() {
        assert_eq!(extract_prefix("", 0), (0, ""));
    }

    #[test]
    fn extract_prefix_at_cursor_walks_back_to_whitespace() {
        let s = "iota(5) + sof";
        let (start, p) = extract_prefix(s, s.len());
        assert_eq!(start, 10);
        assert_eq!(p, "sof");
    }

    #[test]
    fn extract_prefix_handles_colon_commands() {
        let (_, p) = extract_prefix(":intro", 6);
        assert_eq!(p, ":intro");
    }

    #[test]
    fn extract_prefix_stops_at_paren() {
        let s = "foo(bar";
        let (start, p) = extract_prefix(s, 7);
        assert_eq!(start, 4);
        assert_eq!(p, "bar");
    }

    #[test]
    fn match_candidates_empty_prefix_yields_empty() {
        assert!(match_candidates("", REPL_COMMANDS).is_empty());
    }

    #[test]
    fn match_candidates_unique_match() {
        let v = match_candidates(":intro", REPL_COMMANDS);
        assert_eq!(v, vec![":introspect".to_string()]);
    }

    #[test]
    fn match_candidates_ambiguous() {
        let v = match_candidates(":v", REPL_COMMANDS);
        assert_eq!(v, vec![":vars".to_string(), ":version".to_string()]);
    }

    #[test]
    fn match_candidates_skips_exact_matches() {
        // The user already typed the full name; no completion
        // needed.
        let v = match_candidates(":vars", REPL_COMMANDS);
        assert!(v.is_empty());
    }

    #[test]
    fn apply_completion_replaces_prefix_at_cursor() {
        let (out, cur) = apply_completion(":intro", 6, ":introspect");
        assert_eq!(out, ":introspect");
        assert_eq!(cur, 11);
    }

    #[test]
    fn apply_completion_preserves_trailing_text() {
        // Cursor in the middle of the input -- text after
        // cursor should survive.
        let (out, cur) = apply_completion("sof + 1", 3, "softmax");
        assert_eq!(out, "softmax + 1");
        assert_eq!(cur, 7);
    }

    #[test]
    fn compute_tab_match_unique_inserts_completion() {
        let builtins = std::iter::empty();
        match super::compute_tab_match(":intro", 6, builtins) {
            super::TabMatch::Apply { input, cursor } => {
                assert_eq!(input, ":introspect");
                assert_eq!(cursor, 11);
            }
            _ => panic!("expected unique match"),
        }
    }

    #[test]
    fn compute_tab_match_ambiguous_returns_popup() {
        let builtins = std::iter::empty();
        match super::compute_tab_match(":v", 2, builtins) {
            super::TabMatch::Popup(v) => {
                assert!(v.contains(&":vars".to_string()));
                assert!(v.contains(&":version".to_string()));
            }
            _ => panic!("expected popup"),
        }
    }

    #[test]
    fn compute_tab_match_includes_runtime_builtins() {
        // Pretend the runtime exposes only "softmax" and
        // "scatter" -- prefix "s" should pull both into the
        // popup.
        let builtins = ["softmax", "scatter"].into_iter();
        match super::compute_tab_match("s", 1, builtins) {
            super::TabMatch::Popup(v) => {
                assert!(v.contains(&"softmax".to_string()));
                assert!(v.contains(&"scatter".to_string()));
            }
            _ => panic!("expected popup with builtin candidates"),
        }
    }

    #[test]
    fn compute_tab_match_empty_prefix_yields_none() {
        let builtins = std::iter::empty();
        assert!(matches!(
            super::compute_tab_match("foo ", 4, builtins),
            super::TabMatch::None
        ));
    }

    #[test]
    fn trigger_predicate_only_fires_on_ctrl_space() {
        use super::is_completion_trigger;
        // ctrl_key=true + code="Space" -> fire.
        assert!(is_completion_trigger(true, "Space"));
        // Plain Space (no ctrl) -> normal keypress, no fire.
        assert!(!is_completion_trigger(false, "Space"));
        // Ctrl+Tab -> Tab is reserved for browser nav.
        assert!(!is_completion_trigger(true, "Tab"));
        // Plain Tab -> reserved for browser; no fire.
        assert!(!is_completion_trigger(false, "Tab"));
        // Layout-independent: `code()` returns "Space" even
        // on non-US keyboards; `key()` would vary.
        assert!(!is_completion_trigger(true, "KeyJ"));
    }
}
