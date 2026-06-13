//! Tests for `indent_source`, the brace/`;`-aware MLPL source indenter
//! used by `:list <fn>` and the 3D step view. Lives as an integration
//! test so the `indent` module stays a facade of small helpers.

use mlpl_eval_core::indent_source;

#[test]
fn indents_blocks_and_breaks_on_semicolons() {
    let flat = "def u:sign(x) { a = 1; if gt(x, 0) { 1 } else { 0 } }";
    let out = indent_source(flat);
    assert!(out.starts_with("def u:sign(x) {\n"), "header: {out}");
    assert!(out.contains("\n    a = 1;"), "stmt at depth 1: {out}");
    assert!(out.contains("\n    if gt(x, 0) {"), "if at depth 1: {out}");
    assert!(out.contains("\n        1"), "then-body at depth 2: {out}");
    assert!(out.contains("} else {"), "else rides the brace line: {out}");
    assert!(
        out.trim_end().ends_with("\n}"),
        "closing brace on its own line: {out}"
    );
}

#[test]
fn does_not_break_inside_a_string_literal() {
    // A `;` or `{` inside a string must be copied verbatim, not treated
    // as structure.
    let out = indent_source(r#"svg("a; b { c }")"#);
    assert_eq!(out, r#"svg("a; b { c }")"#, "string copied verbatim: {out}");
}

#[test]
fn flat_input_without_braces_is_unchanged() {
    assert_eq!(indent_source("a = 1 + 2"), "a = 1 + 2");
}
