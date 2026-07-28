//! Batch-aware (rank-3 input) attention tests. Saga 29 step 008.
//!
//! Confirms the forward and tape paths accept `[B, T, d_model]`
//! input, agree with the per-batch rank-2 path, and that
//! `attention_weights` returns `[B, T, T]` for rank-3 input.
//! Gradcheck confirms the tape lowering is correct on a tiny
//! `[2, 3, 4]` fixture.

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

fn fresh_env_with_model_and_input() -> (Environment, Vec<f64>) {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("mdl = attention(4, 1, 17)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let xs: Vec<f64> = (0..24).map(|i| (i as f64) * 0.01 - 0.1).collect();
    (env, xs)
}

#[test]
fn rank2_input_still_works_unchanged() {
    let (mut env, _) = fresh_env_with_model_and_input();
    env.set(
        "x".into(),
        arr(vec![3, 4], (0..12).map(|i| i as f64).collect()),
    );
    let y = run("apply(mdl, x)", &mut env);
    assert_eq!(y.shape().dims(), &[3, 4]);
}

#[test]
fn rank3_input_returns_b_t_d() {
    let (mut env, xs) = fresh_env_with_model_and_input();
    env.set("x".into(), arr(vec![2, 3, 4], xs));
    let y = run("apply(mdl, x)", &mut env);
    assert_eq!(y.shape().dims(), &[2, 3, 4]);
}

#[test]
fn rank3_per_batch_agrees_with_rank2_calls() {
    let (mut env, xs) = fresh_env_with_model_and_input();
    env.set("x3".into(), arr(vec![2, 3, 4], xs.clone()));
    env.set("x0".into(), arr(vec![3, 4], xs[..12].to_vec()));
    env.set("x1".into(), arr(vec![3, 4], xs[12..].to_vec()));
    let y3 = run("apply(mdl, x3)", &mut env);
    let y0 = run("apply(mdl, x0)", &mut env);
    let y1 = run("apply(mdl, x1)", &mut env);
    assert_eq!(y3.shape().dims(), &[2, 3, 4]);
    for i in 0..12 {
        assert!(
            (y3.data()[i] - y0.data()[i]).abs() < 1e-9,
            "batch 0 entry {i}: rank-3 = {} vs rank-2 = {}",
            y3.data()[i],
            y0.data()[i]
        );
        assert!(
            (y3.data()[12 + i] - y1.data()[i]).abs() < 1e-9,
            "batch 1 entry {i}: rank-3 = {} vs rank-2 = {}",
            y3.data()[12 + i],
            y1.data()[i]
        );
    }
}

#[test]
fn attention_weights_rank3_returns_b_t_t() {
    let (mut env, xs) = fresh_env_with_model_and_input();
    env.set("x".into(), arr(vec![2, 3, 4], xs));
    let y = run("attention_weights(mdl, x)", &mut env);
    assert_eq!(y.shape().dims(), &[2, 3, 3]);
    // Every row should sum to 1.
    for b in 0..2 {
        for r in 0..3 {
            let row_sum: f64 = y.data()[b * 9 + r * 3..b * 9 + r * 3 + 3].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "batch {b} row {r}: sum {row_sum} not 1"
            );
        }
    }
}

#[test]
fn rank3_multi_head_attention_weights_returns_b_heads_t_t() {
    // Saga 29 step 013: rank-3 + multi-head no longer rejects;
    // attention_weights returns [B, heads, T, T] for batched
    // multi-head input.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("mdl_h2 = attention(4, 2, 17)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let xs: Vec<f64> = (0..24).map(|i| (i as f64) * 0.01 - 0.1).collect();
    env.set("x".into(), arr(vec![2, 3, 4], xs));
    let y = run("attention_weights(mdl_h2, x)", &mut env);
    assert_eq!(y.shape().dims(), &[2, 2, 3, 3]);
    // Every per-(batch, head) row should sum to 1.
    for b in 0..2 {
        for h in 0..2 {
            for r in 0..3 {
                let base = ((b * 2 + h) * 3 + r) * 3;
                let row_sum: f64 = y.data()[base..base + 3].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-9,
                    "batch {b} head {h} row {r}: sum {row_sum} not 1"
                );
            }
        }
    }
}

#[test]
fn rank3_apply_grad_matches_finite_difference_on_wq() {
    // Use the model DSL so apply(mdl, X) walks the tape. For a
    // tiny [2, 3, 4] input with attention(4, 1, ...), check that
    // grad(sum(apply(mdl, X)), Wq*) agrees with finite differences
    // on one entry.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("mdl = attention(4, 1, 17)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let xs: Vec<f64> = (0..24).map(|i| (i as f64) * 0.01 - 0.1).collect();
    env.set("X".into(), arr(vec![2, 3, 4], xs.clone()));
    // mdl's params are named like attention_0_Wq, ... Find one
    // to differentiate against by reading from the model_params
    // namespace. Easier: use the model name -- adam(loss, mdl)
    // pattern walks all params, but grad(loss, name) needs a
    // specific param name.
    let names = mlpl_eval::model_params(&env, "mdl").expect("model params");
    let wq_name = names
        .iter()
        .find(|n| n.contains("Wq"))
        .expect("Wq param")
        .clone();

    let grad_src = format!("grad(sum(apply(mdl, X)), {wq_name})");
    let g = run(&grad_src, &mut env);
    assert_eq!(g.shape().dims(), &[4, 4]);

    // Finite-difference check on Wq entry [0, 0]:
    let h = 1e-4_f64;
    let base = env.get(&wq_name).unwrap().clone();
    let mut plus = base.data().to_vec();
    plus[0] += h;
    env.set(
        wq_name.clone(),
        DenseArray::new(base.shape().clone(), plus).unwrap(),
    );
    let l_plus: f64 = run("reduce_add(apply(mdl, X))", &mut env).data()[0];
    let mut minus = base.data().to_vec();
    minus[0] -= h;
    env.set(
        wq_name.clone(),
        DenseArray::new(base.shape().clone(), minus).unwrap(),
    );
    let l_minus: f64 = run("reduce_add(apply(mdl, X))", &mut env).data()[0];
    let fd = (l_plus - l_minus) / (2.0 * h);
    let analytic = g.data()[0];
    assert!(
        (analytic - fd).abs() < 1e-4,
        "analytic Wq[0, 0] grad {analytic} disagreed with finite-difference {fd}"
    );
}

#[test]
fn rank3_b2_t4_d8_single_head_path_pins_shape_and_one_element() {
    // Saga 30 step 003 regression guard: pin a specific
    // [B=2, T=4, d_model=8] shape through attention(8, 1) and
    // verify (a) output shape is [B, T, d_model] and (b) one
    // element matches the rank-2 reference computation. The
    // rank-3 path runs the [T, d_model] attention per-batch
    // and stacks via Tensor::stack -- this test pins that
    // behavior so any future regression to a chained binary
    // concat lowering (which would still give the same output
    // but cost O(B^2) tape nodes) fails loudly. The test
    // verifies correctness, not the tape-node count; the cost
    // distinction lives in the doc comment on attention_tape_rank3.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("mdl = attention(8, 1, 17)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let xs: Vec<f64> = (0..(2 * 4 * 8))
        .map(|i| (i as f64) * 0.005 - 0.15)
        .collect();
    env.set("X3".into(), arr(vec![2, 4, 8], xs.clone()));
    env.set("X_b0".into(), arr(vec![4, 8], xs[..32].to_vec()));
    env.set("X_b1".into(), arr(vec![4, 8], xs[32..].to_vec()));
    let y3 = run("apply(mdl, X3)", &mut env);
    let y_b0 = run("apply(mdl, X_b0)", &mut env);
    let y_b1 = run("apply(mdl, X_b1)", &mut env);
    assert_eq!(y3.shape().dims(), &[2, 4, 8]);
    // Spot-check: batch 0 element (1, 3) of rank-3 output must
    // equal rank-2 output at (1, 3). Flat index = 1*8 + 3 = 11.
    let r3_b0_t1_d3 = y3.data()[11];
    let r2_b0_t1_d3 = y_b0.data()[11];
    assert!(
        (r3_b0_t1_d3 - r2_b0_t1_d3).abs() < 1e-9,
        "rank-3 batch-0 [1, 3] = {r3_b0_t1_d3} disagreed with rank-2 [1, 3] = {r2_b0_t1_d3}"
    );
    // And a spot-check on batch 1 at element (2, 5). Flat = 2*8+5 = 21.
    let r3_b1_t2_d5 = y3.data()[32 + 21];
    let r2_b1_t2_d5 = y_b1.data()[21];
    assert!(
        (r3_b1_t2_d5 - r2_b1_t2_d5).abs() < 1e-9,
        "rank-3 batch-1 [2, 5] = {r3_b1_t2_d5} disagreed with rank-2 [2, 5] = {r2_b1_t2_d5}"
    );
}
