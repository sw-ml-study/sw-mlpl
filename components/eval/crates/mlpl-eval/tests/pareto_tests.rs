//! `pareto_front(P, dirs)` -- the [n] mask of non-dominated rows
//! of an [n, k] metric matrix, with dirs[d] = 1 to maximize
//! column d and -1 to minimize it. Composes with compress /
//! scatter_labeled (the frontier substrate of the
//! experiment-quality design).

use mlpl_array::DenseArray;
use mlpl_eval::Environment;

fn eval(src: &str) -> Result<DenseArray, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    let mut env = Environment::new();
    mlpl_eval::eval_program(&stmts, &mut env).map_err(|e| e.to_string())
}

#[test]
fn maximize_both_keeps_the_non_dominated_rows() {
    let m = eval("pareto_front([[1, 1], [2, 2], [0, 3]], [1, 1])").unwrap();
    assert_eq!(m.data(), &[0.0, 1.0, 1.0]);
}

#[test]
fn minimize_both_flips_the_dominance() {
    let m = eval("pareto_front([[1, 1], [2, 2], [0, 3]], [-1, -1])").unwrap();
    assert_eq!(m.data(), &[1.0, 0.0, 1.0]);
}

#[test]
fn mixed_directions_quality_vs_cost() {
    // (quality UP, cost DOWN): (3, 10) beats (2, 12); (1, 2) is the
    // cheap end of the frontier; (2, 12) is strictly worse than (3, 10).
    let m = eval("pareto_front([[3, 10], [2, 12], [1, 2]], [1, -1])").unwrap();
    assert_eq!(m.data(), &[1.0, 0.0, 1.0]);
}

#[test]
fn duplicate_rows_are_both_kept() {
    let m = eval("pareto_front([[1, 1], [1, 1]], [1, 1])").unwrap();
    assert_eq!(m.data(), &[1.0, 1.0]);
}

#[test]
fn composes_with_compress() {
    let f = eval("P = [[1, 1], [2, 2], [0, 3]]; compress(pareto_front(P, [1, 1]), P)").unwrap();
    assert_eq!(f.shape().dims(), &[2, 2]);
    assert_eq!(f.data(), &[2.0, 2.0, 0.0, 3.0]);
}

#[test]
fn bad_dirs_get_tutoring_errors() {
    let e = eval("pareto_front([[1, 1]], [1])").unwrap_err();
    assert!(e.contains("one direction per column"), "{e}");
    let e = eval("pareto_front([[1, 1]], [1, 2])").unwrap_err();
    assert!(e.contains("1 (maximize) or -1 (minimize)"), "{e}");
    let e = eval("pareto_front([1, 2], [1, 1])").unwrap_err();
    assert!(e.contains("rank-2"), "{e}");
}
