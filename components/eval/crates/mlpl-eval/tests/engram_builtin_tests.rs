//! `ngram_hash` / `gather_rows` builtins (saga E1 step 2). The
//! hash output is pinned to mlpl-engram-core's golden_fixture_v1,
//! so the LANGUAGE surface and the cross-backend contract can
//! never drift apart.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> Result<mlpl_array::DenseArray, mlpl_eval::EvalError> {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env)
}

#[test]
fn ngram_hash_matches_the_frozen_fixture() {
    let mut env = Environment::new();
    let out = eval(&mut env, "ngram_hash([10, 20, 30, 40], [2, 3], 4, 1024, 7)").unwrap();
    assert_eq!(out.shape().dims(), &[4, 2, 4], "[T, order, head]");
    let expected: Vec<f64> = [
        587u64, 896, 82, 144, 311, 59, 156, 649, 183, 409, 308, 800, 622, 425, 600, 521, 804, 946,
        534, 432, 503, 360, 946, 842, 401, 460, 760, 65, 383, 297, 268, 141,
    ]
    .iter()
    .map(|&v| v as f64)
    .collect();
    assert_eq!(
        out.data(),
        &expected[..],
        "golden_fixture_v1 via the builtin"
    );
}

#[test]
fn ngram_hash_rejects_fractional_ids_and_bad_arity() {
    let mut env = Environment::new();
    assert!(eval(&mut env, "ngram_hash([1.5], [2], 4, 1024, 7)").is_err());
    assert!(eval(&mut env, "ngram_hash([1], [2], 4, 1024)").is_err());
}

#[test]
fn gather_rows_selects_and_shapes() {
    let mut env = Environment::new();
    let out = eval(
        &mut env,
        "t = reshape(range(8), [4, 2]); gather_rows(t, [[3, 0], [1, 1]])",
    )
    .unwrap();
    assert_eq!(out.shape().dims(), &[2, 2, 2], "indices shape + [dim]");
    assert_eq!(out.data(), &[6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0]);
}

#[test]
fn gather_rows_bounds_are_loud() {
    let mut env = Environment::new();
    let err = eval(&mut env, "gather_rows(reshape(range(4), [2, 2]), [5])").unwrap_err();
    assert!(format!("{err}").contains("out of range"), "{err}");
}

#[test]
fn flatten_ravels_and_deprecated_iota_still_evaluates() {
    // flatten(a): ravel to rank-1 row-major (naming policy: the
    // meaningful name is canonical; APL heritage names are
    // aliases only). iota stays a DEPRECATED alias of range so
    // existing user scripts keep working, but it appears nowhere
    // in docs, demos, or the catalog anymore.
    let mut env = Environment::new();
    let flat = eval(&mut env, "flatten(reshape(range(6), [2, 3]))").unwrap();
    assert_eq!(flat.shape().dims(), &[6]);
    assert_eq!(flat.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let legacy = eval(&mut env, "iota(4)").unwrap();
    assert_eq!(legacy.data(), &[0.0, 1.0, 2.0, 3.0]);
}
