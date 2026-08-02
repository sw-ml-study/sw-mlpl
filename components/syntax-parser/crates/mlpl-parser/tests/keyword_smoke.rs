//! Keyword + construct smoke test.
//!
//! Walks every language keyword, syntactic form, and curated
//! BuiltinRef and asserts each lexes + parses cleanly. Catches
//! parser regressions before they reach a demo. Each entry is
//! the smallest fragment that exercises the construct.

use mlpl_parser::{lex, parse};

const FRAGMENTS: &[(&str, &str)] = &[
    // Numeric / array / string literals
    ("int literal", "42"),
    ("float literal", "3.14"),
    ("negative int", "-7"),
    ("string literal", "\"hello\""),
    ("array literal", "[1, 2, 3]"),
    ("nested array", "[[1, 2], [3, 4]]"),
    // Identifiers and assignment
    ("ident reference", "x = 1\nx"),
    ("multi-char ident", "my_var = 1\nmy_var"),
    // Arithmetic + comparison ops
    ("add", "1 + 2"),
    ("sub", "5 - 2"),
    ("mul", "3 * 4"),
    ("div", "10 / 3"),
    ("paren precedence", "(2 + 3) * 4"),
    ("unary neg", "-x"),
    // Function calls
    ("zero-arg call", "range(5)"),
    ("multi-arg call", "matmul(a, b)"),
    ("nested call", "softmax(matmul(X, W), 1)"),
    // Constructors
    ("param ctor", "W = param[2, 3]"),
    ("tensor ctor", "T = tensor[4]"),
    // Assignment annotation form
    ("labeled annotation", "x : [batch, dim] = randn(0, [2, 3])"),
    // BuiltinRef forms (Saga post-v0.19)
    ("BuiltinRef ident", ":add"),
    ("BuiltinRef long ident", ":sigmoid"),
    ("BuiltinRef plus", ":+"),
    ("BuiltinRef star", ":*"),
    ("BuiltinRef slash", ":/"),
    ("BuiltinRef minus", ":-"),
    ("BuiltinRef in arg", "reduce(:add, [1, 2, 3])"),
    ("BuiltinRef as binding", "f = :max\nf"),
    // Loops + scoped blocks
    ("repeat", "repeat 5 { x = x + 1 }"),
    (
        "train",
        "train 3 { adam(loss, W, 0.01, 0.9, 0.999, 0.00000001) }",
    ),
    ("for in", "for row in [[1], [2]] { row }"),
    (
        "experiment block",
        "experiment \"run1\" { loss_metric = 0.1 }",
    ),
    ("device block", "device(\"cpu\") { y = matmul(X, W) }"),
    // Comments
    ("end-of-line comment", "1 + 2 # trailing"),
    ("comment-only line", "# nothing\nx = 1"),
    // User-defined functions
    ("def", "def u:area(r) { r * r }"),
    ("def multi-param", "def u:add(a, b) { a + b }"),
    (
        "def with return",
        "def u:safe(x) { if gt(x, 0) { return x } else { return 0 } }",
    ),
    ("return bare", "def u:noop() { return }"),
    // Statement separators
    ("semicolon", "a = 1; b = 2; a + b"),
    ("newline", "a = 1\nb = 2\na + b"),
];

#[test]
fn every_language_fragment_parses() {
    let mut failures: Vec<String> = Vec::new();
    for (label, src) in FRAGMENTS {
        match lex(src).and_then(|toks| parse(&toks)) {
            Ok(stmts) if stmts.is_empty() => {
                failures.push(format!("{label}: parsed to zero statements"));
            }
            Ok(_) => {}
            Err(e) => {
                failures.push(format!("{label}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} of {} language fragments failed to parse:\n  - {}",
            failures.len(),
            FRAGMENTS.len(),
            failures.join("\n  - ")
        );
    }
}

#[test]
fn fragments_are_unique() {
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    for (label, _) in FRAGMENTS {
        assert!(seen.insert(label), "duplicate fragment label: {label}");
    }
}
