//! `disp(x)` structural box-diagram builtin (APL2 staging plan,
//! Stage 1). Renders any value as an ASCII box that makes its rank,
//! shape, and depth visible at a glance -- MLPL's answer to APL's
//! `]display`. ASCII-first: the frame uses `+ - |` only.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn disp(src: &str) -> String {
    let mut env = Environment::new();
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    // `disp` returns a `Value::Str`, so use the value-preserving entry.
    eval_program_value(&stmts, &mut env).unwrap().to_string()
}

#[test]
fn disp_scalar() {
    let expected = "\
+---+
| 5 |
+---+
rank 0  shape []  depth 0";
    assert_eq!(disp("disp(5)"), expected);
}

#[test]
fn disp_vector() {
    let expected = "\
+-------+
| 1 2 3 |
+-------+
rank 1  shape [3]  depth 1";
    assert_eq!(disp("disp([1, 2, 3])"), expected);
}

#[test]
fn disp_matrix() {
    let expected = "\
+-------+
| 0 1 2 |
| 3 4 5 |
+-------+
rank 2  shape [2, 3]  depth 1";
    assert_eq!(disp("disp(reshape(iota(6), [2, 3]))"), expected);
}

#[test]
fn disp_rank3_boxes_slices_side_by_side() {
    // A rank-3 tensor renders as ONE ROW of boxed matrices inside
    // an outer frame (the APL2 DISPLAY shape for enclosed blocks),
    // with the shape/rank/depth footer.
    let out = disp("disp(reshape(iota(12), [2, 2, 3]))");
    assert!(out.contains("| 0 1 2 |"), "first block row missing:\n{out}");
    assert!(
        out.contains("|  9 10 11 |"),
        "second block row (block-aligned cells) missing:\n{out}"
    );
    let both = out
        .lines()
        .any(|l| l.contains("0 1 2") && l.contains('6') && l.contains('8'));
    assert!(both, "blocks must sit side by side:\n{out}");
    assert!(
        out.ends_with("rank 3  shape [2, 2, 3]  depth 1"),
        "footer wrong:\n{out}"
    );
}
