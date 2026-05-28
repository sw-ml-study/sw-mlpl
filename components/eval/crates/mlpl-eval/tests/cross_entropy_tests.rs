//! Saga 13 step 004: cross-entropy loss over integer targets.
//!
//! `cross_entropy(logits, targets)` returns a scalar mean
//! negative log-likelihood. Logits are `[N, V]` (or `[B, T, V]`),
//! targets are `[N]` (or `[B, T]`) integer-valued. The op must be
//! numerically stable (large logits must not produce inf/nan),
//! fully differentiable wrt `logits` through `grad(...)`, and
//! produce a clear `ShapeMismatch` error for wrong-shape targets.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn expected_loss(logits: &[Vec<f64>], targets: &[usize]) -> f64 {
    let n = logits.len();
    let mut total = 0.0;
    for (row, &t) in logits.iter().zip(targets.iter()) {
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse: f64 = m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
        total += lse - row[t];
    }
    total / n as f64
}

#[test]
fn cross_entropy_matches_hand_computed_on_4_by_3_example() {
    let mut env = Environment::new();
    let logits = vec![
        vec![1.0, 2.0, 0.5],
        vec![-1.0, 0.0, 2.0],
        vec![0.3, 0.3, 0.3],
        vec![3.0, -2.0, 1.0],
    ];
    let flat: Vec<f64> = logits.iter().flatten().copied().collect();
    env.set("L".into(), arr(vec![4, 3], flat));
    env.set("T".into(), DenseArray::from_vec(vec![1.0, 2.0, 0.0, 0.0]));
    let out = run("cross_entropy(L, T)", &mut env);
    assert_eq!(out.rank(), 0, "cross_entropy returns a scalar");
    let targets = [1usize, 2, 0, 0];
    let want = expected_loss(&logits, &targets);
    let got = out.data()[0];
    assert!(
        (got - want).abs() < 1e-6,
        "cross_entropy mismatch: got {got}, want {want}"
    );
}

#[test]
fn cross_entropy_is_finite_for_very_large_logits() {
    // Max-subtraction must keep log-softmax finite even when logits
    // are O(1e3). Without it, exp(1000) would overflow to +inf.
    let mut env = Environment::new();
    env.set(
        "L".into(),
        arr(vec![2, 3], vec![1e3, 1e3 + 1.0, 1e3 - 1.0, 0.0, 0.0, 5e2]),
    );
    env.set("T".into(), DenseArray::from_vec(vec![1.0, 2.0]));
    let out = run("cross_entropy(L, T)", &mut env);
    let v = out.data()[0];
    assert!(
        v.is_finite(),
        "loss must be finite for large logits, got {v}"
    );
    assert!(v >= 0.0, "nll loss is non-negative, got {v}");
}

#[test]
fn cross_entropy_supports_rank3_logits_and_rank2_targets() {
    // [B=2, T=2, V=3] logits with [B=2, T=2] targets produces a scalar.
    let mut env = Environment::new();
    let data: Vec<f64> = (0..12).map(|i| (i as f64) * 0.1).collect();
    env.set("L".into(), arr(vec![2, 2, 3], data.clone()));
    env.set("T".into(), arr(vec![2, 2], vec![0.0, 1.0, 2.0, 1.0]));
    let out = run("cross_entropy(L, T)", &mut env);
    assert_eq!(out.rank(), 0);

    // Flattened [B*T, V] = [4, 3] with targets [0, 1, 2, 1]. The two
    // formulations should agree.
    let mut env2 = Environment::new();
    env2.set("L".into(), arr(vec![4, 3], data));
    env2.set("T".into(), DenseArray::from_vec(vec![0.0, 1.0, 2.0, 1.0]));
    let flat = run("cross_entropy(L, T)", &mut env2);
    let a = out.data()[0];
    let b = flat.data()[0];
    assert!(
        (a - b).abs() < 1e-12,
        "rank-3 and rank-2 flattened forms disagree: {a} vs {b}"
    );
}

#[test]
fn cross_entropy_wrong_shape_targets_errors_cleanly() {
    // logits [4, 3], targets [3]: should be a ShapeMismatch, not a panic.
    let mut env = Environment::new();
    env.set("L".into(), arr(vec![4, 3], vec![0.0; 12]));
    env.set("T".into(), DenseArray::from_vec(vec![0.0, 1.0, 2.0]));
    let result = eval_program(
        &parse(&lex("cross_entropy(L, T)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "mismatched shapes must error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("cross_entropy"),
        "error should name cross_entropy, got: {msg}"
    );
}

#[test]
fn cross_entropy_out_of_range_target_errors_cleanly() {
    let mut env = Environment::new();
    env.set("L".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.set("T".into(), DenseArray::from_vec(vec![0.0, 5.0]));
    let result = eval_program(
        &parse(&lex("cross_entropy(L, T)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "out-of-range target must error");
}

#[test]
fn grad_of_cross_entropy_wrt_logits_matches_finite_difference() {
    // Analytic: d loss / d logits[i, c] = (softmax(logits)[i, c] -
    // (1 if c == targets[i] else 0)) / N. Compare grad(...) entry to a
    // central finite difference.
    let mut env = Environment::new();
    let setup = "\
        L = param[3, 4]\n\
        T = [0.0, 2.0, 1.0]\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();
    // Seed L with non-zero values so softmax is non-uniform.
    env.set(
        "L".into(),
        arr(
            vec![3, 4],
            vec![
                0.5, -0.2, 0.1, 0.3, //
                -0.4, 0.8, 0.0, 0.6, //
                0.2, 0.3, -0.7, 0.1,
            ],
        ),
    );

    let g = run("grad(cross_entropy(L, T), L)", &mut env);
    assert_eq!(g.shape().dims(), &[3, 4]);
    assert!(g.data().iter().any(|v| v.abs() > 1e-6));

    // Finite-difference check on entry [1, 2].
    let h = 1e-4_f64;
    let base = env.get("L").unwrap().clone();
    let idx = 4 + 2;
    let mut plus = base.data().to_vec();
    plus[idx] += h;
    env.set(
        "L".into(),
        DenseArray::new(base.shape().clone(), plus).unwrap(),
    );
    let l_plus = run("cross_entropy(L, T)", &mut env).data()[0];
    let mut minus = base.data().to_vec();
    minus[idx] -= h;
    env.set(
        "L".into(),
        DenseArray::new(base.shape().clone(), minus).unwrap(),
    );
    let l_minus = run("cross_entropy(L, T)", &mut env).data()[0];
    let fd = (l_plus - l_minus) / (2.0 * h);
    let analytic = g.data()[idx];
    assert!(
        (analytic - fd).abs() < 1e-5,
        "analytic grad {analytic} disagreed with finite-difference {fd}"
    );
}

#[test]
fn grad_of_cross_entropy_sums_to_zero_per_row() {
    // For any row i, sum_c (softmax[i, c] - one_hot[i, c]) = 0, so
    // the per-row gradient must sum to zero after dividing by N.
    let mut env = Environment::new();
    let setup = "\
        L = param[3, 4]\n\
        T = [0.0, 3.0, 2.0]\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();
    env.set(
        "L".into(),
        arr(
            vec![3, 4],
            vec![
                0.5, -0.2, 0.1, 0.3, -0.4, 0.8, 0.0, 0.6, 0.2, 0.3, -0.7, 0.1,
            ],
        ),
    );
    let g = run("grad(cross_entropy(L, T), L)", &mut env);
    for i in 0..3 {
        let s: f64 = g.data()[i * 4..(i + 1) * 4].iter().sum();
        assert!(s.abs() < 1e-12, "row {i} gradient sum is {s}, expected 0");
    }
}

#[test]
fn cross_entropy_can_drive_adam_training_loop() {
    // Smoke: a tiny classifier can be trained against cross_entropy
    // via grad + adam and see its loss drop.
    let mut env = Environment::new();
    // Note: `adam` must receive the live loss expression, not a
    // variable bound to its previously-evaluated scalar value --
    // otherwise the param `W` is not on the computation graph and
    // grad is zero. This mirrors how adam is wired in every other
    // training demo in this repo.
    // Include a bias column so the three-class decision boundary is
    // expressible -- without it, the "class 0 covers (1,0) and (-1,0)
    // but class 2 covers (1,1)" data is still separable but slower to
    // fit in the small iteration budget. Added the third feature
    // column as a constant 1.
    let src = "\
        W = param[3, 3]\n\
        X = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]]\n\
        T = [0.0, 1.0, 2.0, 0.0]\n\
        initial = cross_entropy(matmul(X, W), T)\n\
        train 150 {\n\
            adam(cross_entropy(matmul(X, W), T), W, 0.2, 0.9, 0.999, 0.00000001)\n\
            cross_entropy(matmul(X, W), T)\n\
        }\n\
        final = cross_entropy(matmul(X, W), T)\n";
    eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env).unwrap();
    let initial = env.get("initial").unwrap().data()[0];
    let final_ = env.get("final").unwrap().data()[0];
    assert!(
        final_ < initial * 0.5,
        "cross_entropy loss did not drop enough: initial={initial}, final={final_}"
    );
}
