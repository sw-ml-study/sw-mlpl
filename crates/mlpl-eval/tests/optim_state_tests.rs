//! Saga 10, step 001: optimizer state scaffolding tests.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, OptimizerState, eval_program, optim_state, optim_state_mut};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn key(o: &str, p: &str, s: &str) -> (String, String, String) {
    (o.into(), p.into(), s.into())
}

#[test]
fn fresh_state_has_no_buffers_and_zero_steps() {
    let s = OptimizerState::default();
    assert!(s.buffers.is_empty());
    assert!(s.steps.is_empty());
}

#[test]
fn buffers_round_trip_through_environment() {
    let mut env = Environment::new();
    let v = arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    optim_state_mut(&mut env)
        .buffers
        .insert(key("momentum_sgd", "W", "v"), v.clone());
    let got = optim_state(&env)
        .buffers
        .get(&key("momentum_sgd", "W", "v"))
        .unwrap();
    assert_eq!(got.data(), v.data());
    // Different optimizer name = different slot, no collision.
    assert!(
        !optim_state(&env)
            .buffers
            .contains_key(&key("adam", "W", "v"))
    );
}

#[test]
fn step_counters_are_independent_per_optimizer() {
    let mut env = Environment::new();
    let steps = &mut optim_state_mut(&mut env).steps;
    *steps.entry("adam".into()).or_insert(0) += 1;
    *steps.entry("adam".into()).or_insert(0) += 1;
    *steps.entry("momentum_sgd".into()).or_insert(0) += 1;
    let s = optim_state(&env);
    assert_eq!(s.steps.get("adam").copied().unwrap_or(0), 2);
    assert_eq!(s.steps.get("momentum_sgd").copied().unwrap_or(0), 1);
}

#[test]
fn adam_step_one_matches_reference_with_bias_correction() {
    // W=[2.0], loss=sum(W*W). At step t=1:
    //   g    = 4.0
    //   m    = 0.1*4.0     = 0.4
    //   v    = 0.001*16.0  = 0.016
    //   mhat = 0.4/(1-0.9) = 4.0
    //   vhat = 0.016/0.001 = 16.0
    //   step = 0.1 * 4.0 / (sqrt(16) + 1e-8) ~= 0.1
    //   W'   ~= 1.9
    let mut env = Environment::new();
    env.set_param("W".into(), arr(vec![1], vec![2.0]));
    let stmts = parse(&lex("adam(sum(W*W), W, 0.1, 0.9, 0.999, 0.00000001)").unwrap()).unwrap();
    eval_program(&stmts, &mut env).unwrap();

    let m = optim_state(&env)
        .buffers
        .get(&key("adam", "W", "m"))
        .expect("adam m buffer present");
    let v = optim_state(&env)
        .buffers
        .get(&key("adam", "W", "v"))
        .expect("adam v buffer present");
    assert!((m.data()[0] - 0.4).abs() < 1e-9);
    assert!((v.data()[0] - 0.016).abs() < 1e-9);
    assert_eq!(optim_state(&env).steps.get("adam").copied(), Some(1));

    let w = env.get("W").unwrap().data()[0];
    assert!((w - 1.9).abs() < 1e-6, "expected W ~= 1.9, got {w}");
}

#[test]
fn adam_beats_sgd_on_poorly_conditioned_quadratic() {
    // loss = 100*X*X + Y*Y. Hessian condition number 100; vanilla SGD
    // with a single learning rate either diverges on X or crawls on Y,
    // while Adam's per-coordinate scaling handles both.
    let src = "100*sum(X*X) + sum(Y*Y)";
    let n = 200;

    // Adam run
    let mut env_a = Environment::new();
    env_a.set_param("X".into(), arr(vec![1], vec![1.0]));
    env_a.set_param("Y".into(), arr(vec![1], vec![1.0]));
    let stmts_a =
        parse(&lex(&format!("adam({src}, [X, Y], 0.1, 0.9, 0.999, 0.00000001)")).unwrap()).unwrap();
    for _ in 0..n {
        eval_program(&stmts_a, &mut env_a).unwrap();
    }
    let xa = env_a.get("X").unwrap().data()[0];
    let ya = env_a.get("Y").unwrap().data()[0];
    let adam_norm = (xa * xa + ya * ya).sqrt();

    // Plain SGD run via momentum_sgd with beta=0
    let mut env_s = Environment::new();
    env_s.set_param("X".into(), arr(vec![1], vec![1.0]));
    env_s.set_param("Y".into(), arr(vec![1], vec![1.0]));
    let stmts_s =
        parse(&lex(&format!("momentum_sgd({src}, [X, Y], 0.005, 0.0)")).unwrap()).unwrap();
    for _ in 0..n {
        eval_program(&stmts_s, &mut env_s).unwrap();
    }
    let xs = env_s.get("X").unwrap().data()[0];
    let ys = env_s.get("Y").unwrap().data()[0];
    let sgd_norm = (xs * xs + ys * ys).sqrt();

    assert!(
        adam_norm < sgd_norm,
        "expected Adam to outperform SGD: adam_norm={adam_norm}, sgd_norm={sgd_norm}"
    );
    assert!(
        adam_norm < 0.1,
        "expected Adam to be near optimum, got norm={adam_norm}"
    );
}

#[test]
fn momentum_sgd_converges_on_quadratic() {
    // loss = sum(W*W); minimum at W=0. Starting from W=[2.0], 100 steps
    // of momentum-SGD with lr=0.1, beta=0.9 should drive W close to 0.
    let mut env = Environment::new();
    env.set_param("W".into(), arr(vec![1], vec![2.0]));
    let stmts = parse(&lex("momentum_sgd(sum(W*W), W, 0.1, 0.9)").unwrap()).unwrap();
    for _ in 0..100 {
        eval_program(&stmts, &mut env).unwrap();
    }
    let w = env.get("W").unwrap();
    assert!(
        w.data()[0].abs() < 1e-2,
        "expected W ~= 0 after 100 steps, got {}",
        w.data()[0]
    );
}

#[test]
fn momentum_sgd_velocity_persists_between_calls() {
    // Single step on loss = sum(W*W), W=[2.0]:
    //   grad = 2W = [4.0]
    //   v_1  = 0.9*0 + 4.0 = 4.0
    //   v_2  = 0.9*4.0 + 2*W_1 where W_1 = 2 - 0.1*4 = 1.6 -> grad=3.2
    //          v_2 = 3.6 + 3.2 = 6.8
    let mut env = Environment::new();
    env.set_param("W".into(), arr(vec![1], vec![2.0]));
    let stmts = parse(&lex("momentum_sgd(sum(W*W), W, 0.1, 0.9)").unwrap()).unwrap();

    eval_program(&stmts, &mut env).unwrap();
    let v1 = optim_state(&env)
        .buffers
        .get(&key("momentum_sgd", "W", "v"))
        .expect("velocity buffer present after first call");
    assert!((v1.data()[0] - 4.0).abs() < 1e-9);
    assert_eq!(
        optim_state(&env).steps.get("momentum_sgd").copied(),
        Some(1)
    );

    eval_program(&stmts, &mut env).unwrap();
    let v2 = optim_state(&env)
        .buffers
        .get(&key("momentum_sgd", "W", "v"))
        .expect("velocity buffer present after second call");
    assert!(
        (v2.data()[0] - 6.8).abs() < 1e-9,
        "expected v_2 = 6.8, got {}",
        v2.data()[0]
    );
    assert_eq!(
        optim_state(&env).steps.get("momentum_sgd").copied(),
        Some(2)
    );
}

#[test]
fn momentum_sgd_supports_param_list() {
    // Two params, list form: momentum_sgd(loss, [A, B], lr, beta).
    let mut env = Environment::new();
    env.set_param("A".into(), arr(vec![1], vec![1.0]));
    env.set_param("B".into(), arr(vec![1], vec![-2.0]));
    let stmts =
        parse(&lex("momentum_sgd(sum(A*A) + sum(B*B), [A, B], 0.1, 0.0)").unwrap()).unwrap();
    eval_program(&stmts, &mut env).unwrap();
    // beta=0, so v=grad. A_new = 1 - 0.1*2 = 0.8, B_new = -2 - 0.1*-4 = -1.6
    assert!((env.get("A").unwrap().data()[0] - 0.8).abs() < 1e-9);
    assert!((env.get("B").unwrap().data()[0] + 1.6).abs() < 1e-9);
}
