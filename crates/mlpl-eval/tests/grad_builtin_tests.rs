//! Tests for the `grad(expr, wrt)` built-in.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

#[test]
fn grad_scalar_loss_wrt_vector_param() {
    // loss = sum(w * w); d loss / d w = 2 * w
    let mut env = Environment::new();
    env.set_param("w".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    let g = run("grad(sum(w * w), w)", &mut env);
    assert_eq!(g.shape(), &Shape::vector(3));
    assert_eq!(g.data(), &[2.0, 4.0, 6.0]);
}

#[test]
fn grad_matrix_param_via_sum() {
    // loss = sum(W * W); d loss / d W = 2 * W
    let mut env = Environment::new();
    let w = DenseArray::new(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    env.set_param("W".into(), w);
    let g = run("grad(sum(W * W), W)", &mut env);
    assert_eq!(g.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(g.data(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn grad_zero_grad_reset_between_calls() {
    // Two successive grad calls on the same param must each return
    // the current (un-accumulated) gradient.
    let mut env = Environment::new();
    env.set_param("w".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    let g1 = run("grad(sum(w * w), w)", &mut env);
    let g2 = run("grad(sum(w * w), w)", &mut env);
    assert_eq!(g1.data(), g2.data());
    assert_eq!(g1.data(), &[2.0, 4.0, 6.0]);
}

#[test]
fn grad_param_from_ctor_is_tracked() {
    // Params introduced via `w = param[3]` should be tracked even
    // without calling set_param explicitly.
    let mut env = Environment::new();
    let src = "w = param[3]\ng = grad(sum(w * w + w), w)\ng";
    let g = run(src, &mut env);
    // w starts at zeros, so d(sum(w*w + w))/dw = 2w + 1 = [1, 1, 1]
    assert_eq!(g.shape(), &Shape::vector(3));
    assert_eq!(g.data(), &[1.0, 1.0, 1.0]);
}
