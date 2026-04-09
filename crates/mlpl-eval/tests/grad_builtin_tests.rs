//! Tests for the `grad(expr, wrt)` built-in.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, model_params};
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
fn grad_through_apply_linear_matches_hand_rolled() {
    // A linear layer's forward pass is XW + 1@b. grad(loss_via_apply, W)
    // must match grad of the hand-written equivalent element-for-element.
    let mut env = Environment::new();
    let setup = "\
        mdl = linear(2, 2, 7)\n\
        X = [[1.0, 0.5], [-0.5, 0.25], [0.75, -1.0]]\n\
        Y = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]\n\
        O = ones([3, 1])\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();

    let names = model_params(&env, "mdl").expect("mdl is a model");
    let w_name = names[0].clone();
    let b_name = names[1].clone();

    let apply_src = format!("grad(mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)), {w_name})");
    let hand_src = format!(
        "grad(mean((matmul(X, {w_name}) + matmul(O, {b_name}) - Y) * \
         (matmul(X, {w_name}) + matmul(O, {b_name}) - Y)), {w_name})"
    );
    let g_apply = run(&apply_src, &mut env);
    let g_hand = run(&hand_src, &mut env);
    assert_eq!(g_apply.shape(), g_hand.shape());
    for (a, b) in g_apply.data().iter().zip(g_hand.data().iter()) {
        assert!((a - b).abs() < 1e-10, "grad mismatch: {a} vs {b}");
    }
    // Gradient must be non-trivial (not all zeros) for the test to be meaningful.
    assert!(g_apply.data().iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn grad_through_apply_chain_linear_tanh_linear() {
    // chain(linear, tanh_layer, linear) forward pass wired onto the tape.
    // Gradient wrt the outer W2 must match the hand-rolled tanh MLP form.
    let mut env = Environment::new();
    let setup = "\
        mdl = chain(linear(2, 3, 11), tanh_layer(), linear(3, 2, 13))\n\
        X = [[0.8, -0.2], [0.1, 0.9]]\n\
        Y = [[1.0, 0.0], [0.0, 1.0]]\n\
        O2 = ones([2, 1])\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();

    // chain params are W1,b1,W2,b2 in declaration order.
    let names = model_params(&env, "mdl").expect("mdl is a model");
    assert_eq!(names.len(), 4);
    let w1 = names[0].clone();
    let b1 = names[1].clone();
    let w2 = names[2].clone();
    let b2 = names[3].clone();

    let apply_src = format!("grad(mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)), {w2})");
    let hand_src = format!(
        "grad(mean((matmul(tanh(matmul(X, {w1}) + matmul(O2, {b1})), {w2}) + matmul(O2, {b2}) - Y) * \
         (matmul(tanh(matmul(X, {w1}) + matmul(O2, {b1})), {w2}) + matmul(O2, {b2}) - Y)), {w2})"
    );
    let g_apply = run(&apply_src, &mut env);
    let g_hand = run(&hand_src, &mut env);
    assert_eq!(g_apply.shape(), g_hand.shape());
    for (a, b) in g_apply.data().iter().zip(g_hand.data().iter()) {
        assert!((a - b).abs() < 1e-10, "grad mismatch: {a} vs {b}");
    }
    assert!(g_apply.data().iter().any(|v| v.abs() > 1e-6));
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
