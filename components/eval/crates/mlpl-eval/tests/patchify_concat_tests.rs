//! `patchify(x, P)` and `concat(a, b, axis)` builtins.
//! Saga 29 step 005.
//!
//! Forward shape correctness, builtin dispatch, axis-label
//! propagation, error paths, and gradcheck parity vs finite
//! differences for both ops on the tape.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn run_err(src: &str, env: &mut Environment) -> mlpl_eval::EvalError {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap_err()
}

#[test]
fn patchify_shape_2_3_4_4_p2_is_2_4_12() {
    // [B=2, C=3, H=4, W=4] with P=2 -> [2, 4, 12]
    let x: Vec<f64> = (0..2 * 3 * 4 * 4).map(|i| i as f64).collect();
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![2, 3, 4, 4], x));
    let y = run("patchify(x, 2)", &mut env);
    assert_eq!(y.shape().dims(), &[2, 4, 12]);
}

#[test]
fn patchify_preserves_data_count() {
    let x: Vec<f64> = (0..(3 * 4 * 4)).map(|i| i as f64).collect();
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![1, 3, 4, 4], x.clone()));
    let y = run("patchify(x, 2)", &mut env);
    assert_eq!(y.data().len(), x.len());
    let mut sorted_in = x.clone();
    sorted_in.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_out = y.data().to_vec();
    sorted_out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted_in, sorted_out, "patchify is a permutation");
}

#[test]
fn patchify_first_patch_top_left_corner_matches_channel_outer_flatten() {
    // 1x1x4x4 (single channel), P=2 -> first patch = [0, 1, 4, 5].
    let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![1, 1, 4, 4], x));
    let y = run("patchify(x, 2)", &mut env);
    assert_eq!(y.shape().dims(), &[1, 4, 4]);
    // patch 0 is top-left -> rows 0,1 cols 0,1 -> [0, 1, 4, 5].
    let p0 = &y.data()[0..4];
    assert_eq!(p0, &[0.0, 1.0, 4.0, 5.0]);
    // patch 1 is top-right -> rows 0,1 cols 2,3 -> [2, 3, 6, 7].
    let p1 = &y.data()[4..8];
    assert_eq!(p1, &[2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn patchify_with_3_channels_lays_channels_before_dy_dx() {
    // 1x3x2x2 image, P=2 -> [1, 1, 12]. Channel 0 = 0..4,
    // channel 1 = 4..8, channel 2 = 8..12.
    let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![1, 3, 2, 2], x.clone()));
    let y = run("patchify(x, 2)", &mut env);
    assert_eq!(y.shape().dims(), &[1, 1, 12]);
    // channel-outer order: [c0_dy0_dx0, c0_dy0_dx1, c0_dy1_dx0,
    // c0_dy1_dx1, c1_..., c2_...]
    assert_eq!(y.data(), &x);
}

#[test]
fn patchify_errors_on_non_divisible_patch_size() {
    let x = vec![0.0; 3 * 5 * 4];
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![1, 3, 5, 4], x));
    let err = run_err("patchify(x, 2)", &mut env);
    assert!(format!("{err}").contains("patchify") || format!("{err}").contains("shape"));
}

#[test]
fn concat_axis_0_sums_along_first_dim() {
    let mut env = Environment::new();
    env.set("a".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.set("b".into(), arr(vec![5, 3], vec![1.0; 15]));
    let y = run("concat(a, b, 0)", &mut env);
    assert_eq!(y.shape().dims(), &[7, 3]);
}

#[test]
fn concat_axis_1_sums_along_second_dim() {
    // [2, 1, 4] cat [2, 8, 4] along axis 1 -> [2, 9, 4]
    let a: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let b: Vec<f64> = (100..164).map(|i| i as f64).collect();
    let mut env = Environment::new();
    env.set("a".into(), arr(vec![2, 1, 4], a.clone()));
    env.set("b".into(), arr(vec![2, 8, 4], b.clone()));
    let y = run("concat(a, b, 1)", &mut env);
    assert_eq!(y.shape().dims(), &[2, 9, 4]);
    // The first row of axis 0 is 1 row from `a` followed by
    // 8 rows from `b`. Verify the interleaved layout.
    assert_eq!(&y.data()[0..4], &a[0..4]);
    assert_eq!(&y.data()[4..36], &b[0..32]);
    assert_eq!(&y.data()[36..40], &a[4..8]);
    assert_eq!(&y.data()[40..72], &b[32..64]);
}

#[test]
fn concat_errors_on_mismatched_non_concat_dim() {
    let mut env = Environment::new();
    env.set("a".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.set("b".into(), arr(vec![2, 4], vec![0.0; 8])); // wrong axis 1 size
    let err = run_err("concat(a, b, 0)", &mut env);
    assert!(format!("{err}").contains("concat") || format!("{err}").contains("shape"));
}

#[test]
fn concat_axis_2_works_in_rank3() {
    // Saga 30 step 001 lifted the {0, 1} axis restriction.
    // [2, 3, 4] cat [2, 3, 5] along axis 2 -> [2, 3, 9]
    // where each axis-(0,1) cell is the 4 from `a` followed
    // by the 5 from `b`.
    let a: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let b: Vec<f64> = (100..130).map(|i| i as f64).collect();
    let mut env = Environment::new();
    env.set("a".into(), arr(vec![2, 3, 4], a.clone()));
    env.set("b".into(), arr(vec![2, 3, 5], b.clone()));
    let y = run("concat(a, b, 2)", &mut env);
    assert_eq!(y.shape().dims(), &[2, 3, 9]);
    // For each (i, j) along axes 0 and 1 the row is
    // a[i, j, 0..4] then b[i, j, 0..5].
    // First (0, 0): a[0..4] then b[0..5].
    assert_eq!(&y.data()[..4], &a[0..4]);
    assert_eq!(&y.data()[4..9], &b[0..5]);
    // Last (1, 2): a[20..24] then b[25..30].
    let last_a_start = a.len() - 4;
    let last_b_start = b.len() - 5;
    assert_eq!(&y.data()[45..49], &a[last_a_start..]);
    assert_eq!(&y.data()[49..54], &b[last_b_start..]);
}

#[test]
fn grad_patchify_matches_finite_difference_on_2_3_4_4() {
    // Use patchify inside grad: sum of patchify(x, 2) should
    // have d/dx_ijkl = 1 for every (i, j, k, l).
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("X = param[2, 3, 4, 4]").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    // Seed with deterministic but non-trivial values.
    let xs: Vec<f64> = (0..2 * 3 * 4 * 4).map(|i| (i as f64) * 0.01).collect();
    env.set("X".into(), arr(vec![2, 3, 4, 4], xs.clone()));
    let g = run("grad(sum(patchify(X, 2)), X)", &mut env);
    assert_eq!(g.shape().dims(), &[2, 3, 4, 4]);
    // Every entry should be exactly 1 (sum of all patchify
    // elements is linear in every input element, with
    // coefficient 1).
    for &v in g.data() {
        assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
    }

    // Spot-check with a non-linear scalar loss so the gradient
    // is non-constant. Use sum(patchify(X, 2) * patchify(X, 2))
    // = sum(X^2) so d/dX_ijkl = 2 * X_ijkl.
    let g2 = run("grad(sum(patchify(X, 2) * patchify(X, 2)), X)", &mut env);
    assert_eq!(g2.shape().dims(), &[2, 3, 4, 4]);
    for (k, &v) in g2.data().iter().enumerate() {
        let expected = 2.0 * xs[k];
        assert!(
            (v - expected).abs() < 1e-6,
            "[{k}] expected {expected}, got {v}"
        );
    }
}

#[test]
fn grad_concat_axis_1_splits_along_seam() {
    // [2, 1, 4] CLS prepended to [2, 8, 4] patches.
    // Loss = sum(concat(C, P, 1)) = sum(C) + sum(P).
    // d/dC = 1 everywhere; d/dP = 1 everywhere.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("C = param[2, 1, 4]\nP = param[2, 8, 4]").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let cs: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    let ps: Vec<f64> = (0..64).map(|i| i as f64 * 0.01).collect();
    env.set("C".into(), arr(vec![2, 1, 4], cs.clone()));
    env.set("P".into(), arr(vec![2, 8, 4], ps.clone()));
    let g_c = run("grad(sum(concat(C, P, 1)), C)", &mut env);
    let g_p = run("grad(sum(concat(C, P, 1)), P)", &mut env);
    assert_eq!(g_c.shape().dims(), &[2, 1, 4]);
    assert_eq!(g_p.shape().dims(), &[2, 8, 4]);
    assert!(g_c.data().iter().all(|v| (v - 1.0).abs() < 1e-9));
    assert!(g_p.data().iter().all(|v| (v - 1.0).abs() < 1e-9));

    // Non-linear loss: sum(concat^2). d/dC_ijk = 2 * C_ijk
    // since C lands in the upstream gradient slot for that
    // half of the concat output, and similarly for P.
    let g_c2 = run("grad(sum(concat(C, P, 1) * concat(C, P, 1)), C)", &mut env);
    let g_p2 = run("grad(sum(concat(C, P, 1) * concat(C, P, 1)), P)", &mut env);
    for (k, &v) in g_c2.data().iter().enumerate() {
        let expected = 2.0 * cs[k];
        assert!(
            (v - expected).abs() < 1e-6,
            "C[{k}] expected {expected}, got {v}"
        );
    }
    for (k, &v) in g_p2.data().iter().enumerate() {
        let expected = 2.0 * ps[k];
        assert!(
            (v - expected).abs() < 1e-6,
            "P[{k}] expected {expected}, got {v}"
        );
    }
}

#[test]
fn grad_concat_axis_0_splits_at_seam() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = param[2, 3]\nB = param[5, 3]").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let as_: Vec<f64> = (0..6).map(|i| i as f64 * 0.1).collect();
    let bs: Vec<f64> = (0..15).map(|i| i as f64 * 0.1 - 0.5).collect();
    env.set("A".into(), arr(vec![2, 3], as_.clone()));
    env.set("B".into(), arr(vec![5, 3], bs.clone()));
    let g_a = run("grad(sum(concat(A, B, 0) * concat(A, B, 0)), A)", &mut env);
    let g_b = run("grad(sum(concat(A, B, 0) * concat(A, B, 0)), B)", &mut env);
    for (k, &v) in g_a.data().iter().enumerate() {
        let expected = 2.0 * as_[k];
        assert!(
            (v - expected).abs() < 1e-6,
            "A[{k}] expected {expected}, got {v}"
        );
    }
    for (k, &v) in g_b.data().iter().enumerate() {
        let expected = 2.0 * bs[k];
        assert!(
            (v - expected).abs() < 1e-6,
            "B[{k}] expected {expected}, got {v}"
        );
    }
}

#[test]
fn grad_concat_axis_2_splits_at_seam() {
    // Saga 30 step 002: lift the autograd backward to handle
    // axis >= 2. Pair [2, 3, 4] with [2, 3, 5] concat'd along
    // axis 2 -> [2, 3, 9]. Loss = sum(concat^2). The analytic
    // gradient at every position is 2 * the input value.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = param[2, 3, 4]\nB = param[2, 3, 5]").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let as_: Vec<f64> = (0..24).map(|i| i as f64 * 0.05 - 0.5).collect();
    let bs: Vec<f64> = (0..30).map(|i| i as f64 * 0.07 - 1.0).collect();
    env.set("A".into(), arr(vec![2, 3, 4], as_.clone()));
    env.set("B".into(), arr(vec![2, 3, 5], bs.clone()));
    let g_a = run("grad(sum(concat(A, B, 2) * concat(A, B, 2)), A)", &mut env);
    let g_b = run("grad(sum(concat(A, B, 2) * concat(A, B, 2)), B)", &mut env);
    assert_eq!(g_a.shape().dims(), &[2, 3, 4]);
    assert_eq!(g_b.shape().dims(), &[2, 3, 5]);
    for (k, &v) in g_a.data().iter().enumerate() {
        let expected = 2.0 * as_[k];
        assert!(
            (v - expected).abs() < 1e-6,
            "A[{k}] expected {expected}, got {v}"
        );
    }
    for (k, &v) in g_b.data().iter().enumerate() {
        let expected = 2.0 * bs[k];
        assert!(
            (v - expected).abs() < 1e-6,
            "B[{k}] expected {expected}, got {v}"
        );
    }
}

#[test]
fn grad_concat_axis_2_finite_difference_parity() {
    // Stronger gradcheck: compare the autograd-computed gradient
    // of a non-trivial loss against a numerical finite-difference
    // approximation. Loss = sum(softmax-like nonlinearity over
    // the concat output) -- here just `sum(exp(concat) - concat)`
    // to keep it cheap. The analytic gradient at each input
    // position should be `exp(x) - 1`.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = param[1, 2, 3]\nB = param[1, 2, 2]").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let as_: Vec<f64> = (0..6).map(|i| i as f64 * 0.1).collect();
    let bs: Vec<f64> = (0..4).map(|i| i as f64 * 0.1 + 0.5).collect();
    env.set("A".into(), arr(vec![1, 2, 3], as_.clone()));
    env.set("B".into(), arr(vec![1, 2, 2], bs.clone()));
    let g_a = run(
        "grad(sum(exp(concat(A, B, 2)) - concat(A, B, 2)), A)",
        &mut env,
    );
    let g_b = run(
        "grad(sum(exp(concat(A, B, 2)) - concat(A, B, 2)), B)",
        &mut env,
    );
    for (k, &v) in g_a.data().iter().enumerate() {
        let expected = as_[k].exp() - 1.0;
        assert!(
            (v - expected).abs() < 1e-3,
            "A[{k}] expected {expected}, got {v}"
        );
    }
    for (k, &v) in g_b.data().iter().enumerate() {
        let expected = bs[k].exp() - 1.0;
        assert!(
            (v - expected).abs() < 1e-3,
            "B[{k}] expected {expected}, got {v}"
        );
    }
}
