//! RS2 (../reasoning-from-scratch): softmax(a, axis) is axis-aware and
//! differentiable inside grad, for rank up to 3 (attention needs a
//! non-last axis and batched rows).

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

fn fresh(src: &str) -> DenseArray {
    let mut env = Environment::new();
    run(src, &mut env)
}

#[test]
fn eager_softmax_axis_sums_to_one_along_axis() {
    // 2x3; softmax over axis 0 -> each column sums to 1.
    let out = fresh("softmax(reshape(range(6), [2, 3]), 0)");
    assert_eq!(out.shape().dims(), &[2, 3]);
    for c in 0..3 {
        let s: f64 = (0..2).map(|r| out.data()[r * 3 + c]).sum();
        assert!((s - 1.0).abs() < 1e-9, "column {c} sums to 1");
    }
}

#[test]
fn eager_softmax_rank3_last_axis() {
    // rank-3 softmax over the last axis: 2*3 = 6 lanes each summing to 1.
    let out = fresh("softmax(reshape(range(24), [2, 3, 4]), 2)");
    assert_eq!(out.shape().dims(), &[2, 3, 4]);
    for lane in 0..6 {
        let s: f64 = (0..4).map(|k| out.data()[lane * 4 + k]).sum();
        assert!((s - 1.0).abs() < 1e-9, "lane {lane} sums to 1");
    }
}

#[test]
fn grad_softmax_axis0_matches_finite_differences() {
    let base = [0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6];
    let mut env = Environment::new();
    env.set_param(
        "x".into(),
        DenseArray::new(Shape::new(vec![2, 3]), base.to_vec()).unwrap(),
    );
    // Loss couples all entries so the axis actually matters.
    let loss = "sum(softmax(x, 0) * reshape(range(6), [2, 3]))";
    let g = run(&format!("grad({loss}, x)"), &mut env);
    let eps = 1e-6;
    let w: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let f = |data: &[f64]| -> f64 {
        // softmax over axis 0 (columns of 2x3), dotted with w.
        let mut sm = [0.0f64; 6];
        for c in 0..3 {
            let col = [data[c], data[3 + c]];
            let m = col[0].max(col[1]);
            let e = [(col[0] - m).exp(), (col[1] - m).exp()];
            let s = e[0] + e[1];
            sm[c] = e[0] / s;
            sm[3 + c] = e[1] / s;
        }
        sm.iter().zip(w.iter()).map(|(a, b)| a * b).sum()
    };
    for i in 0..6 {
        let (mut p, mut m) = (base, base);
        p[i] += eps;
        m[i] -= eps;
        let fd = (f(&p) - f(&m)) / (2.0 * eps);
        assert!(
            (g.data()[i] - fd).abs() < 1e-4,
            "fd mismatch at {i}: {} vs {fd}",
            g.data()[i]
        );
    }
}

#[test]
fn grad_softmax_rank3_last_axis_flows() {
    let mut env = Environment::new();
    let data: Vec<f64> = (0..24).map(|n| n as f64 * 0.1).collect();
    env.set_param(
        "x".into(),
        DenseArray::new(Shape::new(vec![2, 3, 4]), data).unwrap(),
    );
    // Non-uniform weights so the gradient is non-trivial.
    let g = run(
        "grad(sum(softmax(x, 2) * reshape(range(24), [2, 3, 4])), x)",
        &mut env,
    );
    assert_eq!(g.shape().dims(), &[2, 3, 4]);
    // softmax Jacobian rows sum to zero -> each lane's grad sums to ~0.
    for lane in 0..6 {
        let s: f64 = (0..4).map(|k| g.data()[lane * 4 + k]).sum();
        assert!(s.abs() < 1e-9, "lane {lane} grad sums to zero, got {s}");
    }
    assert!(
        g.data().iter().any(|v| v.abs() > 1e-6),
        "grad is non-trivial"
    );
}

#[test]
fn default_last_axis_unchanged() {
    // softmax(x) with no axis == softmax over the last axis (rows of 2x3).
    let a = fresh("softmax(reshape(range(6), [2, 3]))");
    let b = fresh("softmax(reshape(range(6), [2, 3]), 1)");
    for i in 0..6 {
        assert!((a.data()[i] - b.data()[i]).abs() < 1e-12, "entry {i}");
    }
}
