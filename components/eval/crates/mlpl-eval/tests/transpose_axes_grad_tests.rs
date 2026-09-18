//! RS3 (../reasoning-from-scratch): transpose_axes(x, perm) -- a general axis
//! permutation -- is differentiable inside grad (backward permutes the
//! gradient by the inverse permutation). Reverse-axes transpose is unchanged.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

// loss = reduce_add(transpose_axes(x, [2,0,1]) * reshape(range(24), [4,2,3]))
const LOSS: &str = "reduce_add(transpose_axes(x, [2, 0, 1]) * reshape(range(24), [4, 2, 3]))";

fn loss_at(env: &mut Environment, data: Vec<f64>) -> f64 {
    env.set_param(
        "x".into(),
        DenseArray::new(Shape::new(vec![2, 3, 4]), data).unwrap(),
    );
    run(&format!("({LOSS})"), env).data()[0]
}

#[test]
fn grad_through_transpose_axes_matches_finite_differences() {
    let base: Vec<f64> = (0..24).map(|n| n as f64 * 0.1).collect();
    let mut env = Environment::new();
    env.set_param(
        "x".into(),
        DenseArray::new(Shape::new(vec![2, 3, 4]), base.clone()).unwrap(),
    );
    let g = run(&format!("grad({LOSS}, x)"), &mut env);
    assert_eq!(g.shape().dims(), &[2, 3, 4]);
    let eps = 1e-6;
    for i in 0..24 {
        let (mut p, mut m) = (base.clone(), base.clone());
        p[i] += eps;
        m[i] -= eps;
        let fd = (loss_at(&mut env, p) - loss_at(&mut env, m)) / (2.0 * eps);
        assert!(
            (g.data()[i] - fd).abs() < 1e-4,
            "fd mismatch at {i}: {} vs {fd}",
            g.data()[i]
        );
    }
    assert!(
        g.data().iter().any(|v| v.abs() > 1e-6),
        "grad is non-trivial"
    );
}

#[test]
fn reverse_transpose_grad_unchanged() {
    // grad(sum(transpose(x)), x) == all ones (transpose is a pure permutation).
    let mut env = Environment::new();
    env.set_param(
        "x".into(),
        DenseArray::new(Shape::new(vec![2, 3]), (0..6).map(|n| n as f64).collect()).unwrap(),
    );
    let g = run("grad(sum(transpose(x)), x)", &mut env);
    assert_eq!(g.shape().dims(), &[2, 3]);
    for v in g.data() {
        assert!((v - 1.0).abs() < 1e-9, "reverse-transpose grad is ones");
    }
}
