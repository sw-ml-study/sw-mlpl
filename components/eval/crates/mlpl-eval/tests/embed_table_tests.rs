//! Saga 16.5 step 002: `embed_table(model)` builtin.
//!
//! Walks a `ModelSpec` tree and returns the first
//! Embedding layer's `[vocab, d_model]` table. Closes
//! the Saga 16 gap where training a full
//! `chain(embed, transformer_block, head)` had no
//! source-level way to pull the learned embedding back
//! out.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(env: &mut Environment, src: &str) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

#[test]
fn embed_table_standalone_returns_table() {
    // `embed_table(embed(V, d, seed))` returns the same
    // rows that `apply(emb, iota(V))` gathers, i.e. the
    // full lookup table itself.
    let mut env = Environment::new();
    run(&mut env, "emb = embed(5, 3, 7)");
    let via_apply = run(&mut env, "apply(emb, iota(5))");
    let via_embed_table = run(&mut env, "embed_table(emb)");
    assert_eq!(via_embed_table.shape().dims(), &[5, 3]);
    assert_eq!(
        via_apply.data(),
        via_embed_table.data(),
        "embed_table should match apply(emb, iota(V))"
    );
}

#[test]
fn embed_table_in_chain_returns_embed_not_linear() {
    // Chain has an embed at position 0 and a linear at
    // position 1. embed_table must walk to the embed and
    // ignore the linear's W.
    let mut env = Environment::new();
    run(&mut env, "m = chain(embed(6, 4, 3), linear(4, 6, 2))");
    let standalone = run(&mut env, "embed_table(embed(6, 4, 3))");
    let from_chain = run(&mut env, "embed_table(m)");
    assert_eq!(from_chain.shape().dims(), &[6, 4]);
    assert_eq!(
        standalone.data(),
        from_chain.data(),
        "chain's embed_table should equal a standalone embed with the same seed"
    );
}

#[test]
fn embed_table_in_residual_recurses() {
    // Residual(embed(...)) is semantically weird at apply
    // time (embed changes shape) but the spec tree is
    // well-formed; embed_table should still walk into
    // the inner node.
    let mut env = Environment::new();
    run(&mut env, "m = residual(embed(4, 2, 1))");
    let standalone = run(&mut env, "embed_table(embed(4, 2, 1))");
    let from_residual = run(&mut env, "embed_table(m)");
    assert_eq!(from_residual.shape().dims(), &[4, 2]);
    assert_eq!(standalone.data(), from_residual.data());
}

#[test]
fn embed_table_nested_chain_walks_down() {
    let mut env = Environment::new();
    run(
        &mut env,
        "m = chain(chain(embed(5, 3, 2), linear(3, 3, 0)), linear(3, 5, 1))",
    );
    let standalone = run(&mut env, "embed_table(embed(5, 3, 2))");
    let from_nested = run(&mut env, "embed_table(m)");
    assert_eq!(from_nested.shape().dims(), &[5, 3]);
    assert_eq!(standalone.data(), from_nested.data());
}

#[test]
fn embed_table_reflects_trained_weights() {
    // After an adam step, the table should change and
    // embed_table must return the updated weights.
    let mut env = Environment::new();
    run(&mut env, "m = chain(embed(4, 2, 5), linear(2, 4, 6))");
    let before = run(&mut env, "embed_table(m)").data().to_vec();
    env.set("X".into(), arr(vec![3], vec![0.0, 1.0, 2.0]));
    env.set(
        "Y".into(),
        arr(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    );
    run(
        &mut env,
        "train 5 { adam(mean((apply(m, X) - Y) * (apply(m, X) - Y)), m, 0.1, 0.9, 0.999, 0.00000001) }",
    );
    let after = run(&mut env, "embed_table(m)").data().to_vec();
    assert_eq!(before.len(), after.len());
    let diff: f64 = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "embed_table must reflect trained weights; diff = {diff}"
    );
}

#[test]
fn embed_table_errors_when_no_embedding() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(3, 4, 0)");
    let stmts = parse(&lex("embed_table(m)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("no embedding -> error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("embed_table") && msg.contains("no embedding"),
        "got: {msg}"
    );
}

#[test]
fn embed_table_errors_on_wrong_arity() {
    let mut env = Environment::new();
    run(&mut env, "e = embed(3, 2, 0)");
    let stmts = parse(&lex("embed_table(e, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("arity should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("embed_table") || msg.contains("arity"),
        "got: {msg}"
    );
}

#[test]
fn embed_table_errors_on_non_model_argument() {
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let stmts = parse(&lex("embed_table(X)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("array-not-model should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("embed_table"), "got: {msg}");
}
