//! Saga 13 step 006: end-to-end tiny LM training demo primitives.
//!
//! This file covers two things:
//!
//! 1. The `shift_pairs_x` / `shift_pairs_y` builtins that slice a
//!    1-D token id array `[N]` into two rank-2 arrays `[B, T]` of
//!    input tokens and next-token labels. The slicing rule is
//!    contiguous, non-overlapping windows of size `block_size + 1`:
//!
//!        B = N / (block_size + 1)  (integer division; trailing
//!                                    tokens that do not fill a full
//!                                    window are dropped)
//!        X[b, t] = ids[b * (block_size + 1) + t]
//!        Y[b, t] = ids[b * (block_size + 1) + t + 1]
//!
//! 2. An end-to-end tiny LM training run that wires together
//!    every Saga 13 primitive: `embed`, `sinusoidal_encoding`,
//!    `causal_attention`, `rms_norm`, `linear`, `cross_entropy`,
//!    and `adam`. The acceptance bound is that the final loss is
//!    at most 60% of the initial loss.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, eval_program_value};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

// -- shift_pairs_x / shift_pairs_y --

#[test]
fn shift_pairs_x_returns_block_sized_rows() {
    // ids = [0..12], block_size = 3 -> window = 4, B = 12 / 4 = 3.
    let mut env = Environment::new();
    env.set(
        "ids".into(),
        DenseArray::from_vec((0..12).map(|i| i as f64).collect()),
    );
    let out = run("shift_pairs_x(ids, 3)", &mut env);
    assert_eq!(out.shape().dims(), &[3, 3]);
    assert_eq!(out.data(), &[0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0]);
}

#[test]
fn shift_pairs_y_returns_next_token_labels() {
    // Same ids, same block_size -- Y is X shifted by one inside each window.
    let mut env = Environment::new();
    env.set(
        "ids".into(),
        DenseArray::from_vec((0..12).map(|i| i as f64).collect()),
    );
    let out = run("shift_pairs_y(ids, 3)", &mut env);
    assert_eq!(out.shape().dims(), &[3, 3]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0]);
}

#[test]
fn shift_pairs_drops_trailing_tokens_that_cannot_fill_a_window() {
    // ids = [0..10], block_size = 3 -> window = 4, B = 10 / 4 = 2.
    // Tokens 8, 9 are dropped.
    let mut env = Environment::new();
    env.set(
        "ids".into(),
        DenseArray::from_vec((0..10).map(|i| i as f64).collect()),
    );
    let out_x = run("shift_pairs_x(ids, 3)", &mut env);
    let out_y = run("shift_pairs_y(ids, 3)", &mut env);
    assert_eq!(out_x.shape().dims(), &[2, 3]);
    assert_eq!(out_y.shape().dims(), &[2, 3]);
    assert_eq!(out_x.data(), &[0.0, 1.0, 2.0, 4.0, 5.0, 6.0]);
    assert_eq!(out_y.data(), &[1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);
}

#[test]
fn shift_pairs_errors_when_ids_too_short() {
    // Fewer ids than block_size + 1 means B = 0: empty batch axis,
    // which has no training signal. We require at least one window.
    let mut env = Environment::new();
    env.set("ids".into(), DenseArray::from_vec(vec![0.0, 1.0, 2.0]));
    let result = eval_program(
        &parse(&lex("shift_pairs_x(ids, 4)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "short ids must surface an error");
}

#[test]
fn shift_pairs_rejects_non_scalar_block_size() {
    let mut env = Environment::new();
    env.set("ids".into(), DenseArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]));
    env.set("bs".into(), arr(vec![2], vec![2.0, 2.0]));
    let result = eval_program(
        &parse(&lex("shift_pairs_x(ids, bs)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "non-scalar block_size must error");
}

#[test]
fn shift_pairs_rejects_rank_gt_1_ids() {
    let mut env = Environment::new();
    env.set(
        "ids".into(),
        arr(vec![2, 4], (0..8).map(|i| i as f64).collect()),
    );
    let result = eval_program(
        &parse(&lex("shift_pairs_x(ids, 2)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "rank > 1 ids must error");
}

// -- end-to-end tiny LM --

#[test]
fn tiny_shakespeare_snippet_is_preloaded() {
    // The corpus the demo uses must be compiled into the web REPL
    // binary as a preloaded corpus so `load_preloaded("...")` works
    // in a WASM context with no filesystem.
    let mut env = Environment::new();
    let val = eval_program_value(
        &parse(&lex("load_preloaded(\"tiny_shakespeare_snippet\")").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    assert!(matches!(val, mlpl_eval::Value::Str(ref s) if !s.is_empty()));
}

#[test]
fn tiny_lm_training_reduces_loss_by_at_least_forty_percent() {
    // Small end-to-end LM: a one-block causal-attention stack on a
    // short synthetic repeat corpus. The acceptance bound is that
    // the final loss is <= 60% of the initial loss. We keep the
    // training budget small so the test stays under a few seconds
    // on CPU.
    let mut env = Environment::new();
    let src = "\
        corpus = \"abcabcabcabcabcabcabcabcabcabcabcabc\"\n\
        ids    = tokenize_bytes(corpus)\n\
        X_all  = shift_pairs_x(ids, 8)\n\
        Y_all  = shift_pairs_y(ids, 8)\n\
        X      = reshape(X_all, [reduce_mul(shape(X_all))])\n\
        Y      = reshape(Y_all, [reduce_mul(shape(Y_all))])\n\
        V      = 256\n\
        d      = 8\n\
        model  = chain(embed(V, d, 0), \
                       residual(chain(rms_norm(d), causal_attention(d, 1, 1))), \
                       rms_norm(d), \
                       linear(d, V, 4))\n\
        train 40 { \
          adam(cross_entropy(apply(model, X), Y), model, 0.05, 0.9, 0.999, 0.00000001); \
          cross_entropy(apply(model, X), Y) \
        }\n\
        last_losses";
    let losses = eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env).unwrap();
    let data = losses.data();
    assert!(data.len() >= 2, "expected a loss history");
    let l0 = data[0];
    let ln = data[data.len() - 1];
    assert!(
        ln <= 0.6 * l0,
        "final loss {ln} must be at most 60% of initial {l0}"
    );
}
