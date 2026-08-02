//! Saga 79: integration tests for mlpl-web-completion. Live in
//! `tests/` so the test fns don't count against the crate's
//! source-tree module function limits.

use mlpl_web_completion::{
    REPL_COMMANDS, TabMatch, apply_completion, compute_tab_match, extract_prefix,
    is_completion_trigger, match_candidates, next_index, prev_index, should_accept_right,
};

#[test]
fn extract_prefix_on_empty_input_returns_empty() {
    assert_eq!(extract_prefix("", 0), (0, ""));
}

#[test]
fn extract_prefix_at_cursor_walks_back_to_whitespace() {
    let s = "range(5) + sof";
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
    let (out, cur) = apply_completion("sof + 1", 3, "softmax");
    assert_eq!(out, "softmax + 1");
    assert_eq!(cur, 7);
}

#[test]
fn compute_tab_match_unique_inserts_completion() {
    let builtins = std::iter::empty();
    match compute_tab_match(":intro", 6, builtins) {
        TabMatch::Apply { input, cursor } => {
            assert_eq!(input, ":introspect");
            assert_eq!(cursor, 11);
        }
        _ => panic!("expected unique match"),
    }
}

#[test]
fn compute_tab_match_ambiguous_returns_popup() {
    let builtins = std::iter::empty();
    match compute_tab_match(":v", 2, builtins) {
        TabMatch::Popup(v) => {
            assert!(v.contains(&":vars".to_string()));
            assert!(v.contains(&":version".to_string()));
        }
        _ => panic!("expected popup"),
    }
}

#[test]
fn compute_tab_match_includes_runtime_builtins() {
    let builtins = ["softmax", "scatter"].into_iter();
    match compute_tab_match("s", 1, builtins) {
        TabMatch::Popup(v) => {
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
        compute_tab_match("foo ", 4, builtins),
        TabMatch::None
    ));
}

#[test]
fn next_index_wraps_at_end() {
    assert_eq!(next_index(0, 3), 1);
    assert_eq!(next_index(2, 3), 0);
    assert_eq!(next_index(0, 1), 0);
    assert_eq!(next_index(0, 0), 0);
}

#[test]
fn prev_index_wraps_at_start() {
    assert_eq!(prev_index(0, 3), 2);
    assert_eq!(prev_index(1, 3), 0);
    assert_eq!(prev_index(0, 1), 0);
    assert_eq!(prev_index(0, 0), 0);
}

#[test]
fn accept_right_requires_popup_and_cursor_at_end() {
    assert!(should_accept_right(true, 5, 5));
    assert!(should_accept_right(true, 6, 5));
    assert!(!should_accept_right(true, 3, 5));
    assert!(!should_accept_right(false, 5, 5));
}

#[test]
fn trigger_predicate_only_fires_on_ctrl_space() {
    assert!(is_completion_trigger(true, "Space"));
    assert!(!is_completion_trigger(false, "Space"));
    assert!(!is_completion_trigger(true, "Tab"));
    assert!(!is_completion_trigger(false, "Tab"));
    assert!(!is_completion_trigger(true, "KeyJ"));
}
