//! Label-aware lowering tests (compile-to-rust step 004).

use mlpl_lower_rs::{LowerError, lower};
use mlpl_parser::{lex, parse};

fn lower_src(src: &str) -> Result<String, LowerError> {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    lower(&stmts).map(|ts| ts.to_string())
}

fn lower_err(src: &str) -> LowerError {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    lower(&stmts).expect_err("expected lower err")
}

// -- Runtime lowering for label builtins --

#[test]
fn label_builtin_lowers_to_with_labels() {
    let s = lower_src("label(iota(5), [\"seq\"])").unwrap();
    assert!(s.contains(". with_labels"), "{s}");
    assert!(s.contains("\"seq\""), "{s}");
}

#[test]
fn relabel_builtin_shares_path() {
    let s = lower_src("relabel(iota(5), [\"time\"])").unwrap();
    assert!(s.contains(". with_labels"), "{s}");
    assert!(s.contains("\"time\""), "{s}");
}

#[test]
fn reshape_labeled_builtin_chains_reshape_and_labels() {
    let s = lower_src("reshape_labeled(iota(6), [2, 3], [\"r\", \"c\"])").unwrap();
    assert!(s.contains(":: mlpl_rt :: reshape"), "{s}");
    assert!(s.contains(". with_labels"), "{s}");
    assert!(s.contains("\"r\""), "{s}");
    assert!(s.contains("\"c\""), "{s}");
}

#[test]
fn annotation_syntax_desugars_through_parser_to_label() {
    // Parser desugars `x : [seq] = expr` -> Assign(x, FnCall("label", [expr, ArrayLit([StrLit(seq)])]))
    let s = lower_src("x : [seq] = iota(5)").unwrap();
    assert!(s.contains("let x ="), "{s}");
    assert!(s.contains(". with_labels"), "{s}");
    assert!(s.contains("\"seq\""), "{s}");
}

#[test]
fn label_with_non_string_labels_errors() {
    // Second arg must be an all-StrLit ArrayLit.
    let err = lower_err("label(iota(3), [1, 2, 3])");
    assert!(
        matches!(err, LowerError::LabelsMustBeStringLiterals(ref n) if n == "label"),
        "got {err:?}"
    );
}

// -- matmul runtime lowering --

#[test]
fn matmul_lowers_to_dense_matmul() {
    let s = lower_src("matmul(reshape(iota(6), [2, 3]), reshape(iota(12), [3, 4]))").unwrap();
    assert!(s.contains(". matmul"), "{s}");
}

// -- Static matmul contraction check --

#[test]
fn matmul_matching_contraction_labels_lowers_fine() {
    // [seq, d] @ [d, heads] is fine -- contraction axis "d" matches.
    let s = lower_src(
        "a : [seq, d] = reshape(iota(6), [2, 3])\n\
         b : [d, heads] = reshape(iota(12), [3, 4])\n\
         matmul(a, b)",
    )
    .unwrap();
    assert!(s.contains(". matmul"), "{s}");
}

#[test]
fn matmul_mismatched_contraction_labels_errors_at_lower_time() {
    let err = lower_err(
        "a : [seq, d] = reshape(iota(6), [2, 3])\n\
         b : [time, heads] = reshape(iota(12), [3, 4])\n\
         matmul(a, b)",
    );
    match err {
        LowerError::StaticShapeMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, "matmul");
            assert_eq!(expected, vec![Some("seq".into()), Some("d".into())]);
            assert_eq!(actual, vec![Some("time".into()), Some("heads".into())]);
        }
        other => panic!("expected StaticShapeMismatch, got {other:?}"),
    }
}

#[test]
fn matmul_unknown_labels_falls_through_to_runtime() {
    // Neither operand has static labels -> no check performed.
    let s = lower_src(
        "a = reshape(iota(6), [2, 3])\n\
         b = reshape(iota(12), [3, 4])\n\
         matmul(a, b)",
    )
    .unwrap();
    assert!(s.contains(". matmul"), "{s}");
}

#[test]
fn matmul_one_side_unlabeled_falls_through_to_runtime() {
    // Only a has labels -- no static check possible.
    let s = lower_src(
        "a : [seq, d] = reshape(iota(6), [2, 3])\n\
         b = reshape(iota(12), [3, 4])\n\
         matmul(a, b)",
    )
    .unwrap();
    assert!(s.contains(". matmul"), "{s}");
}

#[test]
fn transpose_swaps_labels_for_static_tracking() {
    // a : [seq, d] -> transpose(a) : [d, seq]. That means
    // matmul(transpose(a), a) has contraction axis "seq" vs "seq"
    // -- should lower fine.
    let s = lower_src(
        "a : [seq, d] = reshape(iota(6), [2, 3])\n\
         matmul(transpose(a), a)",
    )
    .unwrap();
    assert!(s.contains(". matmul"), "{s}");
}

#[test]
fn transpose_swaps_labels_catches_via_mismatch() {
    // transpose flips [seq, d] to [d, seq]. matmul(a, transpose(a))
    // has contraction axis "d" vs "d" -- still matches. But
    // matmul(a, transpose(b)) where b:[time, feat] has contraction
    // "d" vs "feat" -- mismatch.
    let err = lower_err(
        "a : [seq, d] = reshape(iota(6), [2, 3])\n\
         b : [time, feat] = reshape(iota(6), [2, 3])\n\
         matmul(a, transpose(b))",
    );
    assert!(
        matches!(err, LowerError::StaticShapeMismatch { ref op, .. } if op == "matmul"),
        "got {err:?}"
    );
}

// -- labels() still unsupported (REPL-only string builtin) --

#[test]
fn labels_builtin_is_unsupported_in_compile_path() {
    let err = lower_err("labels(iota(3))");
    assert!(
        matches!(err, LowerError::Unsupported(ref m) if m.contains("labels")),
        "got {err:?}"
    );
}
