//! In-chain Engram forward (saga E3 step 1): a `ModelSpec::Engram`
//! inside a `Chain` receives the chain's ORIGINAL input as its token
//! ids (embed-first chains: the chain input IS the id vector), so
//! `apply(chain(..., e, ...), ids)` matches the manually composed
//! pipeline built from explicit `apply_engram` calls.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> Result<mlpl_array::DenseArray, mlpl_eval::EvalError> {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env)
}

/// Overwrite an engram's memory table with ones so every addressed
/// row carries signal (same trick as `engram_apply_tests`).
fn poke_memory_ones(env: &mut Environment, model: &str) {
    let mem_name = env.get_model(model).unwrap().params()[0].clone();
    let mem = env.get(&mem_name).unwrap();
    let ones =
        mlpl_array::DenseArray::new(mem.shape().clone(), vec![1.0; mem.data().len()]).unwrap();
    env.set(mem_name, ones);
}

/// The Tiny-LM-shaped shared prelude: embed -> attention block ->
/// (engram goes here) -> norm -> head, all seeded and reused so the
/// full chain and the manual composition see the SAME parameters.
const PRELUDE: &str = "\
    emb  = embed(16, 8, 0); \
    att  = residual(chain(rms_norm(8), causal_attention(8, 2, 1))); \
    head = chain(rms_norm(8), linear(8, 16, 4)); \
    e    = engram(8, [2], 2, 64, 4, 7)";

#[test]
fn in_chain_engram_matches_manual_composition() {
    let mut env = Environment::new();
    eval(&mut env, PRELUDE).unwrap();
    poke_memory_ones(&mut env, "e");
    let full = eval(
        &mut env,
        "m = chain(emb, att, e, head); ids = [1, 2, 3]; apply(m, ids)",
    )
    .unwrap();
    let manual = eval(
        &mut env,
        "apply(head, apply_engram(e, apply(att, apply(emb, ids)), ids))",
    )
    .unwrap();
    assert_eq!(
        full.data(),
        manual.data(),
        "in-chain engram must equal the explicit apply_engram pipeline"
    );
    // With a poked memory table the engram must actually change the
    // output, or this equality would be vacuous.
    let without = eval(&mut env, "mw = chain(emb, att, head); apply(mw, ids)").unwrap();
    assert_ne!(
        full.data(),
        without.data(),
        "non-zero memory must influence the in-chain output"
    );
}

#[test]
fn fresh_engram_in_chain_is_exactly_transparent() {
    let mut env = Environment::new();
    eval(&mut env, PRELUDE).unwrap();
    let with = eval(
        &mut env,
        "m = chain(emb, att, e, head); ids = [5, 6, 7]; apply(m, ids)",
    )
    .unwrap();
    let without = eval(&mut env, "mw = chain(emb, att, head); apply(mw, ids)").unwrap();
    assert_eq!(
        with.data(),
        without.data(),
        "a freshly constructed engram (zero memory) must be an exact no-op in a chain"
    );
}

#[test]
fn bare_apply_on_engram_is_a_clear_error() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(8, [2], 2, 64, 4, 7)").unwrap();
    let err = eval(&mut env, "apply(e, [1, 2, 3])").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("token ids"),
        "bare apply must point at the id requirement, got: {msg}"
    );
}

#[test]
fn non_id_chain_input_is_a_clear_error() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(8, [2], 2, 64, 4, 7)").unwrap();
    // Rank-2 chain input: usable as the hidden state but NOT as ids.
    let err = eval(
        &mut env,
        "m = chain(e); apply(m, reshape(range(24), [3, 8]))",
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("engram in chain"),
        "error must name the in-chain engram context, got: {msg}"
    );
    assert!(
        msg.contains("rank-1"),
        "error must explain the id-vector shape requirement, got: {msg}"
    );
}

#[test]
fn fractional_chain_input_errors_without_panicking() {
    let mut env = Environment::new();
    eval(
        &mut env,
        "e = engram(8, [2], 2, 64, 4, 7); m = chain(embed(16, 8, 0), e)",
    )
    .unwrap();
    // The leading embed already rejects fractional ids; the whole
    // chain must surface that as an EvalError, never a panic.
    assert!(eval(&mut env, "apply(m, [1.5, 2.5])").is_err());
}
