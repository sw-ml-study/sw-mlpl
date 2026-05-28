//! Saga 16.5 step 001: `pca(X, k)` builtin.
//!
//! Top-k principal component analysis via power
//! iteration + deflation, wrapped into a single builtin.
//! Returns the centered-and-projected data `[N, k]`, not
//! the components themselves (callers who need the
//! eigenvectors can run the power-iteration composition
//! pattern directly; see Saga 8 tutorial lesson).

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run_expr(env: &mut Environment, src: &str) -> DenseArray {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap()
}

/// Sample variance of a flat vector (mean over elements;
/// caller is responsible for the "variance per column"
/// framing).
fn variance(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

#[test]
fn pca_anisotropic_2d_captures_dominant_direction() {
    // Build a highly anisotropic [60, 2] cloud: points
    // stretched along axis 0 (variance ~4) and
    // compressed along axis 1 (variance ~0.01). PCA to
    // rank 1 should capture the x-axis direction and
    // recover most of the total variance.
    let mut env = Environment::new();
    env.set("raw".into(), {
        // raw = randn(0, [60, 2]) * [[2.0, 0.0], [0.0, 0.1]]
        // Build deterministically: alternate sign per row
        // so the fixture doesn't depend on a specific RNG.
        let mut data = Vec::with_capacity(120);
        for i in 0..60 {
            let t = (i as f64) * 0.1 - 3.0;
            let x = 2.0 * t.sin();
            let y = 0.1 * (t * 3.7).cos();
            data.push(x);
            data.push(y);
        }
        arr(vec![60, 2], data)
    });
    // Original total variance: var(x) + var(y).
    let raw = env.get("raw").unwrap();
    let xs: Vec<f64> = (0..60).map(|i| raw.data()[i * 2]).collect();
    let ys: Vec<f64> = (0..60).map(|i| raw.data()[i * 2 + 1]).collect();
    let total_var = variance(&xs) + variance(&ys);

    let projected = run_expr(&mut env, "pca(raw, 1)");
    assert_eq!(projected.shape().dims(), &[60, 1]);
    let captured_var = variance(projected.data());
    assert!(
        captured_var > 0.8 * total_var,
        "rank-1 PCA should capture > 80% of total variance: \
         captured = {captured_var}, total = {total_var}"
    );
}

#[test]
fn pca_shape_preservation() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(
            vec![10, 5],
            (0..50).map(|i| (i as f64 * 0.07).sin()).collect(),
        ),
    );
    for k in 1..=5_usize {
        let src = format!("pca(X, {k})");
        let out = run_expr(&mut env, &src);
        assert_eq!(
            out.shape().dims(),
            &[10, k],
            "pca(X, {k}) should return [10, {k}]"
        );
    }
}

#[test]
fn pca_k_equal_d_preserves_total_variance() {
    // When k = D, PCA is just a change of basis in a
    // centered frame; total variance is preserved.
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(
            vec![20, 3],
            (0..60).map(|i| (i as f64 * 0.13).sin()).collect(),
        ),
    );
    let raw = env.get("X").unwrap().clone();
    let mut total_var_in = 0.0_f64;
    for col in 0..3 {
        let column: Vec<f64> = (0..20).map(|i| raw.data()[i * 3 + col]).collect();
        total_var_in += variance(&column);
    }
    let projected = run_expr(&mut env, "pca(X, 3)");
    assert_eq!(projected.shape().dims(), &[20, 3]);
    let mut total_var_out = 0.0_f64;
    for col in 0..3 {
        let column: Vec<f64> = (0..20).map(|i| projected.data()[i * 3 + col]).collect();
        total_var_out += variance(&column);
    }
    assert!(
        (total_var_in - total_var_out).abs() < 1e-4,
        "k = D should preserve total variance: in = {total_var_in}, out = {total_var_out}"
    );
}

#[test]
fn pca_is_deterministic() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(
            vec![15, 4],
            (0..60).map(|i| (i as f64 * 0.11).sin()).collect(),
        ),
    );
    let y1 = run_expr(&mut env, "pca(X, 2)");
    let y2 = run_expr(&mut env, "pca(X, 2)");
    assert_eq!(y1.data(), y2.data(), "pca should be deterministic");
}

#[test]
fn pca_rejects_non_rank2() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![8], (0..8).map(|i| i as f64).collect()));
    let stmts = parse(&lex("pca(v, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("rank-1 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("pca") && msg.contains("rank"), "got: {msg}");
}

#[test]
fn pca_rejects_k_zero() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(vec![5, 3], (0..15).map(|i| i as f64).collect()),
    );
    let stmts = parse(&lex("pca(X, 0)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("k=0 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("pca") && msg.contains("k"), "got: {msg}");
}

#[test]
fn pca_rejects_k_greater_than_d() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(vec![5, 3], (0..15).map(|i| i as f64).collect()),
    );
    let stmts = parse(&lex("pca(X, 4)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("k > D should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("pca") && msg.contains("k"), "got: {msg}");
}

#[test]
fn pca_rejects_non_finite_input() {
    let mut env = Environment::new();
    let mut data: Vec<f64> = (0..15).map(|i| i as f64).collect();
    data[3] = f64::NAN;
    env.set("X".into(), arr(vec![5, 3], data));
    let stmts = parse(&lex("pca(X, 2)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("NaN in X should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("pca") && (msg.contains("finite") || msg.contains("nan")),
        "got: {msg}"
    );
}

#[test]
fn pca_rejects_wrong_arity() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(vec![5, 3], (0..15).map(|i| i as f64).collect()),
    );
    let stmts = parse(&lex("pca(X)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("1-arg form should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("pca") || msg.contains("arity"), "got: {msg}");
}
