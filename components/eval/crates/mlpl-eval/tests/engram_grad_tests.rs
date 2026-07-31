//! `apply_engram` differentiation on the tape (saga E2 step 3):
//! numeric gradcheck of the memory-table gradient, scatter-ADD
//! locality (only addressed rows move, duplicates accumulate), and
//! end-to-end training via adam moving the table.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> Result<mlpl_array::DenseArray, mlpl_eval::EvalError> {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env)
}

/// Tiny spec shared by the gradcheck tests: hidden 4, one order-2
/// n-gram, 1 head, 8 slots, head_dim 4 -> memory is [8, 4].
fn tiny_engram(env: &mut Environment) -> String {
    eval(env, "e = engram(4, [2], 1, 8, 4, 7)").unwrap();
    eval(env, "h = reshape(range(12), [3, 4])").unwrap();
    env.get_model("e").unwrap().params()[0].clone()
}

/// Loss used across the tests: mean of the squared forward output.
const LOSS: &str = "mean(apply_engram(e, h, [1, 2, 3]) * apply_engram(e, h, [1, 2, 3]))";

#[test]
fn memory_gradient_matches_finite_differences() {
    let mut env = Environment::new();
    let mem_name = tiny_engram(&mut env);
    let analytic = eval(&mut env, &format!("grad({LOSS}, {mem_name})")).unwrap();
    assert_eq!(
        analytic.shape().dims(),
        &[8, 4],
        "grad shape == memory shape"
    );
    let eps = 1e-5;
    let base = env.get(&mem_name).unwrap().clone();
    for i in 0..base.data().len() {
        let mut plus = base.data().to_vec();
        plus[i] += eps;
        let mut minus = base.data().to_vec();
        minus[i] -= eps;
        env.set(
            mem_name.clone(),
            mlpl_array::DenseArray::new(base.shape().clone(), plus).unwrap(),
        );
        let lp = eval(&mut env, LOSS).unwrap().data()[0];
        env.set(
            mem_name.clone(),
            mlpl_array::DenseArray::new(base.shape().clone(), minus).unwrap(),
        );
        let lm = eval(&mut env, LOSS).unwrap().data()[0];
        env.set(mem_name.clone(), base.clone());
        let numeric = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic.data()[i] - numeric).abs() < 1e-6,
            "memory grad[{i}]: analytic {} vs numeric {numeric}",
            analytic.data()[i]
        );
    }
}

#[test]
fn only_addressed_rows_receive_gradient() {
    let mut env = Environment::new();
    let mem_name = tiny_engram(&mut env);
    // Make the loss sensitive to memory content (non-zero rows).
    let mem = env.get(&mem_name).unwrap().clone();
    let filled = mlpl_array::DenseArray::new(
        mem.shape().clone(),
        (0..mem.data().len())
            .map(|i| 0.1 + i as f64 * 0.01)
            .collect(),
    )
    .unwrap();
    env.set(mem_name.clone(), filled);
    let g = eval(&mut env, &format!("grad({LOSS}, {mem_name})")).unwrap();
    // The addressed rows are ngram_hash([1,2,3], [2], 1, 8, 7).
    let addressed = eval(&mut env, "ngram_hash([1, 2, 3], [2], 1, 8, 7)").unwrap();
    let hot: std::collections::HashSet<usize> =
        addressed.data().iter().map(|&v| v as usize).collect();
    for row in 0..8 {
        let row_grad: f64 = g.data()[row * 4..(row + 1) * 4]
            .iter()
            .map(|v| v.abs())
            .sum();
        if hot.contains(&row) {
            assert!(row_grad > 0.0, "addressed row {row} must receive gradient");
        } else {
            assert_eq!(row_grad, 0.0, "unaddressed row {row} must stay zero");
        }
    }
}

#[test]
fn duplicate_addresses_accumulate() {
    // Two identical bigrams address the same row twice; the row's
    // gradient must be the sum of both contributions. Compare a
    // duplicated-position loss against the single-position loss:
    // sum over t of per-position contributions means dup grad ==
    // 2x the lone-position grad for the shared row.
    // 64 slots so the (PAD, 5) bigram at position 0 does NOT
    // collide with the shared (5, 5) row (rows 37 vs 53 under
    // seed 7; asserted below rather than hard-coded).
    let mut env = Environment::new();
    eval(&mut env, "e = engram(4, [2], 1, 64, 4, 7)").unwrap();
    let mem_name = env.get_model("e").unwrap().params()[0].clone();
    let mem = env.get(&mem_name).unwrap().clone();
    let filled =
        mlpl_array::DenseArray::new(mem.shape().clone(), vec![0.5; mem.data().len()]).unwrap();
    env.set(mem_name.clone(), filled);
    // ids [5, 5, 5]: positions 1 and 2 share the bigram (5, 5) ->
    // identical row; position 0's bigram is (PAD, 5).
    eval(&mut env, "hh = reshape(zeros([8]) + 1, [2, 4])").unwrap();
    let g2 = eval(
        &mut env,
        &format!("grad(sum(apply_engram(e, hh, [5, 5])), {mem_name})"),
    )
    .unwrap();
    eval(&mut env, "h3 = reshape(zeros([12]) + 1, [3, 4])").unwrap();
    let g3 = eval(
        &mut env,
        &format!("grad(sum(apply_engram(e, h3, [5, 5, 5])), {mem_name})"),
    )
    .unwrap();
    // Row addressed by bigram (5,5) appears once in the [5,5] ids
    // and twice in [5,5,5]; its gradient contribution must double.
    let hashes = eval(&mut env, "ngram_hash([5, 5], [2], 1, 64, 7)").unwrap();
    assert_ne!(
        hashes.data()[0],
        hashes.data()[1],
        "test premise: the PAD bigram must not collide with the shared row"
    );
    let row = hashes.data()[1] as usize;
    let g2_row: f64 = g2.data()[row * 4..(row + 1) * 4].iter().sum();
    let g3_row: f64 = g3.data()[row * 4..(row + 1) * 4].iter().sum();
    assert!(g2_row != 0.0, "the shared row must receive gradient");
    assert!(
        (g3_row - 2.0 * g2_row).abs() < 1e-9,
        "duplicate addressing must accumulate: single {g2_row}, doubled {g3_row}"
    );
}

#[test]
fn adam_training_moves_only_addressed_memory_and_reduces_loss() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(4, [2], 1, 8, 4, 7)").unwrap();
    let mem_name = env.get_model("e").unwrap().params()[0].clone();
    let before = env.get(&mem_name).unwrap().clone();
    eval(
        &mut env,
        "h = reshape(range(12), [3, 4]); y = reshape(range(12), [3, 4]) + 5",
    )
    .unwrap();
    let step = "adam(mean((apply_engram(e, h, [1, 2, 3]) - y) * (apply_engram(e, h, [1, 2, 3]) - y)), e, 0.05, 0.9, 0.999, 0.00000001)";
    eval(&mut env, &format!("l0 = mean((apply_engram(e, h, [1, 2, 3]) - y) * (apply_engram(e, h, [1, 2, 3]) - y)); train 30 {{ {step} }}")).unwrap();
    let l0 = eval(&mut env, "l0").unwrap().data()[0];
    let l1 = eval(
        &mut env,
        "mean((apply_engram(e, h, [1, 2, 3]) - y) * (apply_engram(e, h, [1, 2, 3]) - y))",
    )
    .unwrap()
    .data()[0];
    assert!(l1 < l0 * 0.9, "training must reduce the loss: {l0} -> {l1}");
    // The memory table moved, but ONLY in the addressed rows.
    let after = env.get(&mem_name).unwrap().clone();
    let addressed = eval(&mut env, "ngram_hash([1, 2, 3], [2], 1, 8, 7)").unwrap();
    let hot: std::collections::HashSet<usize> =
        addressed.data().iter().map(|&v| v as usize).collect();
    let mut moved = 0;
    for row in 0..8 {
        let delta: f64 = (0..4)
            .map(|j| (after.data()[row * 4 + j] - before.data()[row * 4 + j]).abs())
            .sum();
        if hot.contains(&row) {
            moved += i32::from(delta > 0.0);
        } else {
            assert_eq!(delta, 0.0, "unaddressed row {row} must not move");
        }
    }
    assert!(moved > 0, "at least one addressed row must move");
}
