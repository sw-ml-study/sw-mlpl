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
fn disp_rank3_stacks_slices() {
    // A rank-3 tensor renders as a labeled stack of its leading-axis
    // slices, each a framed 2D block, with a shape/rank/depth footer.
    let out = disp("disp(reshape(iota(12), [2, 2, 3]))");
    assert!(out.contains("[0]"), "slice 0 label missing:\n{out}");
    assert!(out.contains("[1]"), "slice 1 label missing:\n{out}");
    assert!(out.contains("| 0 1 2 |"), "first slice row missing:\n{out}");
    assert!(
        out.contains("| 9 10 11 |"),
        "second slice row missing:\n{out}"
    );
    assert!(
        out.ends_with("rank 3  shape [2, 2, 3]  depth 1"),
        "footer wrong:\n{out}"
    );
}
