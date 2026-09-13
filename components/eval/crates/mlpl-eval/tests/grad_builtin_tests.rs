//! Tests for the `grad(expr, wrt)` built-in.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::env_api::*;
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
fn grad_through_apply_residual_wraps_linear() {
    // residual(linear(...)) forward pass is x + (XW + 1@b). The
    // gradient wrt the wrapped W equals the gradient of (XW + 1@b - Y)^2
    // alone in the non-residual case when Y is shifted by x, so the
    // cleanest equivalence is: grad(residual-loss, W) == grad of the
    // same loss with x added into the target. Easier: compare the
    // residual-wrapped grad against the hand-rolled add form.
    let mut env = Environment::new();
    let setup = "\
        outer = residual(linear(2, 2, 19))\n\
        X = [[0.7, -0.1], [0.2, 0.6]]\n\
        Y = [[1.0, 0.0], [0.0, 1.0]]\n\
        O = ones([2, 1])\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();

    let names = model_params(&env, "outer").expect("outer is a model");
    let w = names[0].clone();
    let b = names[1].clone();

    let apply_src = format!("grad(mean((apply(outer, X) - Y) * (apply(outer, X) - Y)), {w})");
    let hand_src = format!(
        "grad(mean((X + matmul(X, {w}) + matmul(O, {b}) - Y) * \
         (X + matmul(X, {w}) + matmul(O, {b}) - Y)), {w})"
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
fn grad_through_apply_rms_norm_produces_nonzero_grad() {
    // chain(linear, rms_norm, linear) must emit rms_norm on the tape
    // so that gradients flow back through the normalization to the
    // upstream linear's parameters. Numerical equivalence to a
    // hand-rolled form is out of scope here; this test just checks
    // that the gradient has the right shape and is non-trivial.
    let mut env = Environment::new();
    let setup = "\
        mdl = chain(linear(3, 4, 31), rms_norm(4), linear(4, 2, 32))\n\
        X = [[1.0, -0.5, 0.25], [0.2, 0.3, -0.7], [0.4, 0.9, 0.1]]\n\
        Y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();

    let names = model_params(&env, "mdl").expect("mdl is a model");
    // rms_norm has no params, so the chain's params are W1,b1,W2,b2
    assert_eq!(names.len(), 4);
    let w1 = names[0].clone();
    let src = format!("grad(mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)), {w1})");
    let g = run(&src, &mut env);
    assert_eq!(g.shape().dims(), &[3, 4]);
    assert!(g.data().iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn grad_through_apply_single_head_attention_produces_nonzero_grad() {
    // Single-head attention lowers to matmul/transpose/softmax/mul on
    // the tape. Gradient wrt Wq must propagate through the softmax
    // and non-trivially change. Multi-head (heads > 1) tape lowering
    // is deferred: it requires slicing that the autograd tape does
    // not yet support.
    let mut env = Environment::new();
    let setup = "\
        attn = attention(4, 1, 41)\n\
        X = [[1.0, 0.0, 0.5, -0.2], [0.2, 0.8, -0.1, 0.3], [-0.3, 0.4, 0.9, 0.1]]\n\
        Y = [[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.0], [0.2, 0.3, 0.0, 0.1]]\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();

    let names = model_params(&env, "attn").expect("attn is a model");
    // attention has Wq, Wk, Wv, Wo
    assert_eq!(names.len(), 4);
    let wq = names[0].clone();
    let src = format!("grad(mean((apply(attn, X) - Y) * (apply(attn, X) - Y)), {wq})");
    let g = run(&src, &mut env);
    assert_eq!(g.shape().dims(), &[4, 4]);
    assert!(g.data().iter().any(|v| v.abs() > 1e-6));
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

#[test]
fn grad_accepts_tanh_fn_alias_inside_grad() {
    // Surface MLPL spells the elementwise tanh as `tanh_fn`
    // (`tanh` itself is reserved for the `tanh_layer()` model
    // layer). The grad lifter must accept both names so that
    // demos / lessons that train an MLP with `tanh_fn` activation
    // can reach a gradient. Regression for the Moons MLP demo.
    let mut env = Environment::new();
    env.set_param("w".into(), DenseArray::from_vec(vec![0.5, 0.5]));
    let via_tanh = run("grad(sum(tanh(w)), w)", &mut env);
    let via_tanh_fn = run("grad(sum(tanh_fn(w)), w)", &mut env);
    assert_eq!(via_tanh.shape(), &Shape::vector(2));
    assert_eq!(via_tanh.data(), via_tanh_fn.data());
}

#[test]
fn grad_through_windows_accumulates_on_overlap() {
    // windows(x,[3]) over [x0..x4] = [[x0,x1,x2],[x1,x2,x3],[x2,x3,x4]];
    // sum reads x0 once, x1 twice, x2 thrice, x3 twice, x4 once, so the
    // gradient is the overlap count [1,2,3,2,1] -- the scatter-ADD backward.
    let mut env = Environment::new();
    env.set_param(
        "x".into(),
        DenseArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );
    let g = run("grad(sum(windows(x, [3])), x)", &mut env);
    assert_eq!(g.data(), &[1.0, 2.0, 3.0, 2.0, 1.0]);
}

#[test]
fn grad_through_reduce_add_and_reduce_full() {
    // The downstream-flagged case: reduce_add / reduce(:add) are now
    // differentiable inside grad() (a full reduce == sum, gradient ones).
    let mut env = Environment::new();
    env.set_param("x".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    assert_eq!(
        run("grad(reduce_add(x), x)", &mut env).data(),
        &[1.0, 1.0, 1.0]
    );
    assert_eq!(
        run("grad(reduce(:add, x), x)", &mut env).data(),
        &[1.0, 1.0, 1.0]
    );
}

#[test]
fn grad_through_multiaxis_reduce() {
    // reduce over axis 1 of [2,3] -> [2], then sum -> scalar; d/dM = ones.
    let mut env = Environment::new();
    env.set_param(
        "M".into(),
        DenseArray::new(Shape::new(vec![2, 3]), (0..6).map(|i| i as f64).collect()).unwrap(),
    );
    let g = run("grad(sum(reduce(:add, M, [1])), M)", &mut env);
    assert_eq!(g.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_rejects_non_add_reduce() {
    let mut env = Environment::new();
    env.set_param("x".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    let stmts = parse(&lex("grad(reduce(:max, x), x)").unwrap()).unwrap();
    assert!(eval_program(&stmts, &mut env).is_err());
}

#[test]
fn grad_through_elementwise_conv_composes_windows_reduce_broadcast() {
    // The elementwise convolution loss, differentiated end to end:
    // reduce(:add, k * windows(x,[2,2]), [2,3,4]) composes windows-backward
    // (scatter-add), the reduce-backward (broadcast), and the rank-3-kernel
    // broadcast (C2). grad wrt x returns an x-shaped gradient.
    let mut env = Environment::new();
    env.set_param(
        "x".into(),
        DenseArray::new(
            Shape::new(vec![2, 3, 3]),
            (0..18).map(|i| i as f64).collect(),
        )
        .unwrap(),
    );
    let g = run(
        "k = reshape(range(8), [2, 2, 2])\n\
         grad(sum(reduce(:add, k * windows(x, [2, 2]), [2, 3, 4])), x)",
        &mut env,
    );
    assert_eq!(g.shape().dims(), &[2, 3, 3]);
}

#[test]
fn grad_through_flatten() {
    // flatten is reshape-to-1D; grad flows back to the [2,3] param as ones.
    let mut env = Environment::new();
    env.set_param(
        "M".into(),
        DenseArray::new(Shape::new(vec![2, 3]), (0..6).map(|i| i as f64).collect()).unwrap(),
    );
    let g = run("grad(sum(flatten(M)), M)", &mut env);
    assert_eq!(g.shape().dims(), &[2, 3]);
    assert_eq!(g.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

// -- moe finding F2: user-defined functions inside grad() --

#[test]
fn grad_through_user_function_matches_inline() {
    // A loss written as a user function differentiates like the inline form.
    let mut env = Environment::new();
    env.set_param("w".into(), DenseArray::from_vec(vec![3.0, 4.0]));
    let via_fn = run(
        "def u:loss(v) { \"sum of squares\"; s = v * v; sum(s) }\ngrad(u:loss(w), w)",
        &mut env,
    );
    // d/dw sum(w^2) = 2w
    assert_eq!(via_fn.data(), &[6.0, 8.0]);
}

#[test]
fn grad_user_function_keeps_global_params_differentiable() {
    // A u:fn referencing a global param k keeps k differentiable.
    let mut env = Environment::new();
    env.set_param("x".into(), DenseArray::from_vec(vec![1.0, 2.0]));
    env.set_param("k".into(), DenseArray::from_vec(vec![5.0, 7.0]));
    // loss = sum(x * k); d/dk = x
    let g = run(
        "def u:l(v) { \"dot\"; sum(v * k) }\ngrad(u:l(x), k)",
        &mut env,
    );
    assert_eq!(g.data(), &[1.0, 2.0]);
}

#[test]
fn grad_user_function_arity_mismatch_errors() {
    let mut env = Environment::new();
    env.set_param("w".into(), DenseArray::from_vec(vec![1.0]));
    let stmts = parse(&lex("def u:f(a, b) { \"two\"; a } \n grad(u:f(w), w)").unwrap()).unwrap();
    assert!(eval_program(&stmts, &mut env).is_err());
}

// -- moe-microscope finding F4: gather_rows differentiable (scatter-add) ------

#[test]
fn grad_through_gather_rows_scatter_adds_into_addressed_rows() {
    // gather_rows(M, [0, 0, 2]); loss = sum of gathered rows. The backward is
    // a scatter-ADD into the addressed rows: row 0 (addressed twice) gets 2,
    // row 2 gets 1, row 1 (never addressed) gets 0 -- exactly like windows'
    // overlap accumulation. This is what lets a from-scratch addressing /
    // embedding table train.
    let mut env = Environment::new();
    let m = DenseArray::new(Shape::new(vec![3, 2]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    env.set_param("M".into(), m);
    let g = run("grad(sum(gather_rows(M, [0, 0, 2])), M)", &mut env);
    assert_eq!(g.shape(), &Shape::new(vec![3, 2]));
    assert_eq!(g.data(), &[2., 2., 0., 0., 1., 1.]);
}

#[test]
fn grad_through_gather_rows_forward_matches_eager() {
    // The tape forward must equal the eager gather (F1-style consistency):
    // a weighted sum over gathered rows differentiates to the row's weights.
    let mut env = Environment::new();
    let m = DenseArray::new(Shape::new(vec![2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    env.set_param("M".into(), m);
    // loss = sum(gather_rows(M, [1])) = 4+5+6 = 15; grad on row 1 = ones.
    let g = run("grad(sum(gather_rows(M, [1])), M)", &mut env);
    assert_eq!(g.data(), &[0., 0., 0., 1., 1., 1.]);
}

// -- moe-microscope follow-up F5: index/mask builtins as stop-gradient --------
// Inside grad, non-differentiable index/mask builtins (argmax, one_hot, eq,
// gt, lt, argtop_k) are computed from the current forward values and inserted
// as constant leaves, so a top-1 router mask can be built inside the loss.
// Gradient flows through the surrounding differentiable ops, never the mask.

#[test]
fn grad_through_top1_router_mask_flows_to_selected_experts() {
    // R: [2 tokens, 3 experts]; mask = one_hot(argmax(R, 1), 3) selects the
    // top expert per token. loss = sum(mask * R); d loss/d R == mask.
    let mut env = Environment::new();
    let r = DenseArray::new(Shape::new(vec![2, 3]), vec![0.1, 0.9, 0.2, 0.7, 0.1, 0.1]).unwrap();
    env.set_param("R".into(), r);
    let g = run("grad(sum(one_hot(argmax(R, 1), 3) * R), R)", &mut env);
    assert_eq!(g.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(g.data(), &[0., 1., 0., 1., 0., 0.]);
}

#[test]
fn grad_through_comparison_mask_is_stop_gradient() {
    // gt(W, 0) is a constant {0,1} mask; loss = sum(gt-mask * W); grad == mask.
    let mut env = Environment::new();
    env.set_param("W".into(), DenseArray::from_vec(vec![-1.0, 2.0, -3.0, 4.0]));
    let g = run("grad(sum(gt(W, 0.0) * W), W)", &mut env);
    assert_eq!(g.data(), &[0., 1., 0., 1.]);
}
