//! Tape differentiation of the in-chain Engram (saga E3 step 2):
//! grad through `apply(chain(..., e, ...), ids)` equals grad
//! through the manually composed `apply_engram` pipeline,
//! frozen-base training moves only the engram (and only its
//! addressed memory rows), and a finite-difference spot check pins
//! the analytic gradient through the full chain.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> Result<mlpl_array::DenseArray, mlpl_eval::EvalError> {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env)
}

/// Tiny-LM-shaped prelude, hidden 4 so finite differences stay
/// cheap: embed -> attention block -> engram -> norm -> head.
/// Memory is [8, 4] (one order-2 n-gram, 1 head, 8 slots).
fn build(env: &mut Environment) -> String {
    eval(
        env,
        "emb  = embed(16, 4, 0); \
         att  = residual(chain(rms_norm(4), causal_attention(4, 2, 1))); \
         head = chain(rms_norm(4), linear(4, 16, 4)); \
         e    = engram(4, [2], 1, 8, 4, 7); \
         m    = chain(emb, att, e, head); \
         ids  = [1, 2, 3]",
    )
    .unwrap();
    env.get_model("e").unwrap().params()[0].clone()
}

/// Fill the engram memory with small non-zero values so the loss
/// is sensitive to every addressed row.
fn fill_memory(env: &mut Environment, mem_name: &str) {
    let mem = env.get(mem_name).unwrap().clone();
    let filled = mlpl_array::DenseArray::new(
        mem.shape().clone(),
        (0..mem.data().len())
            .map(|i| 0.05 + i as f64 * 0.01)
            .collect(),
    )
    .unwrap();
    env.set(mem_name.to_string(), filled);
}

const CHAIN_LOSS: &str = "mean(apply(m, ids) * apply(m, ids))";
const MANUAL_LOSS: &str = "mean(apply(head, apply_engram(e, apply(att, apply(emb, ids)), ids)) \
     * apply(head, apply_engram(e, apply(att, apply(emb, ids)), ids)))";

#[test]
fn chain_grad_matches_manual_composition_grad() {
    let mut env = Environment::new();
    let mem_name = build(&mut env);
    fill_memory(&mut env, &mem_name);
    let g_chain = eval(&mut env, &format!("grad({CHAIN_LOSS}, {mem_name})")).unwrap();
    let g_manual = eval(&mut env, &format!("grad({MANUAL_LOSS}, {mem_name})")).unwrap();
    assert_eq!(g_chain.shape().dims(), &[8, 4]);
    assert_eq!(
        g_chain.data(),
        g_manual.data(),
        "in-chain grad must equal the explicit apply_engram pipeline grad"
    );
    assert!(
        g_chain.data().iter().any(|&v| v != 0.0),
        "the loss must actually be sensitive to the memory table"
    );
}

#[test]
fn frozen_base_training_moves_only_the_engram() {
    let mut env = Environment::new();
    let mem_name = build(&mut env);
    eval(&mut env, "freeze(emb); freeze(att); freeze(head)").unwrap();
    // Snapshot every base parameter before training.
    let base_names: Vec<String> = ["emb", "att", "head"]
        .iter()
        .flat_map(|m| env.get_model(m).unwrap().params())
        .collect();
    let before: Vec<mlpl_array::DenseArray> = base_names
        .iter()
        .map(|n| env.get(n).unwrap().clone())
        .collect();
    let mem_before = env.get(&mem_name).unwrap().clone();
    let step =
        "adam(mean((apply(m, ids) - y) * (apply(m, ids) - y)), m, 0.05, 0.9, 0.999, 0.00000001)";
    eval(
        &mut env,
        &format!(
            "y = apply(m, ids) + 1; \
             l0 = mean((apply(m, ids) - y) * (apply(m, ids) - y)); \
             train 40 {{ {step} }}"
        ),
    )
    .unwrap();
    let l0 = eval(&mut env, "l0").unwrap().data()[0];
    let l1 = eval(&mut env, "mean((apply(m, ids) - y) * (apply(m, ids) - y))")
        .unwrap()
        .data()[0];
    assert!(
        l1 < l0 * 0.9,
        "engram-only training must reduce the loss: {l0} -> {l1}"
    );
    // Base parameters are bit-identical after training.
    for (name, snap) in base_names.iter().zip(&before) {
        let now = env.get(name).unwrap();
        assert_eq!(
            now.data(),
            snap.data(),
            "frozen base param {name} must not move"
        );
    }
    // Memory moved, but ONLY in the addressed rows.
    let mem_after = env.get(&mem_name).unwrap().clone();
    let addressed = eval(&mut env, "ngram_hash(ids, [2], 1, 8, 7)").unwrap();
    let hot: std::collections::HashSet<usize> =
        addressed.data().iter().map(|&v| v as usize).collect();
    let mut moved = 0;
    for row in 0..8 {
        let delta: f64 = (0..4)
            .map(|j| (mem_after.data()[row * 4 + j] - mem_before.data()[row * 4 + j]).abs())
            .sum();
        if hot.contains(&row) {
            moved += i32::from(delta > 0.0);
        } else {
            assert_eq!(delta, 0.0, "unaddressed row {row} must not move");
        }
    }
    assert!(moved > 0, "at least one addressed row must move");
}

#[test]
fn chain_memory_gradient_matches_finite_differences_on_addressed_row() {
    let mut env = Environment::new();
    let mem_name = build(&mut env);
    fill_memory(&mut env, &mem_name);
    let analytic = eval(&mut env, &format!("grad({CHAIN_LOSS}, {mem_name})")).unwrap();
    let addressed = eval(&mut env, "ngram_hash(ids, [2], 1, 8, 7)").unwrap();
    let row = addressed.data()[1] as usize;
    let base = env.get(&mem_name).unwrap().clone();
    let eps = 1e-5;
    for col in 0..4 {
        let i = row * 4 + col;
        let mut plus = base.data().to_vec();
        plus[i] += eps;
        let mut minus = base.data().to_vec();
        minus[i] -= eps;
        env.set(
            mem_name.clone(),
            mlpl_array::DenseArray::new(base.shape().clone(), plus).unwrap(),
        );
        let lp = eval(&mut env, CHAIN_LOSS).unwrap().data()[0];
        env.set(
            mem_name.clone(),
            mlpl_array::DenseArray::new(base.shape().clone(), minus).unwrap(),
        );
        let lm = eval(&mut env, CHAIN_LOSS).unwrap().data()[0];
        env.set(mem_name.clone(), base.clone());
        let numeric = (lp - lm) / (2.0 * eps);
        assert!(
            (analytic.data()[i] - numeric).abs() < 1e-5,
            "memory grad[{row},{col}]: analytic {} vs numeric {numeric}",
            analytic.data()[i]
        );
    }
}

#[test]
fn bare_engram_in_grad_is_a_clear_error() {
    let mut env = Environment::new();
    let mem_name = build(&mut env);
    let err = eval(
        &mut env,
        &format!("grad(mean(apply(e, ids) * apply(e, ids)), {mem_name})"),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("token ids"),
        "bare engram on the tape must point at the id requirement, got: {msg}"
    );
}
