//! gen_clone / gen_reset / gen_stats / multi-row gen_append.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

const MODEL: &str = "m = chain(embed(16, 8, 0), \
    residual(chain(rms_norm(8), causal_attention(8, 2, 1))), \
    rms_norm(8), linear(8, 16, 4))";

fn env_with_state() -> Environment {
    let mut env = Environment::new();
    eval_value(&mut env, MODEL).unwrap();
    eval_value(&mut env, "gs = gen_state(m, [0, 1, 2])").unwrap();
    env
}

#[test]
fn stats_report_the_cache_accounting() {
    let mut env = env_with_state();
    // 3-token prompt, 1 attention layer.
    assert_eq!(scalar(&mut env, "gen_stats(gs).tokens"), 3.0);
    assert_eq!(scalar(&mut env, "gen_stats(gs).layers"), 1.0);
    assert_eq!(scalar(&mut env, "gen_stats(gs).kv_rows"), 3.0);
    // kv_values = layers * rows * d_model (K) * 2 (K and V) = 1*3*8*2 = 48
    assert_eq!(scalar(&mut env, "gen_stats(gs).kv_values"), 48.0);
}

#[test]
fn clone_is_independent() {
    let mut env = env_with_state();
    eval_value(&mut env, "gs2 = gen_clone(gs)").unwrap();
    // Advance the clone only.
    eval_value(&mut env, "gen_append(gs2, 5)").unwrap();
    assert_eq!(scalar(&mut env, "gen_stats(gs2).tokens"), 4.0);
    assert_eq!(
        scalar(&mut env, "gen_stats(gs).tokens"),
        3.0,
        "original untouched"
    );
}

#[test]
fn reset_returns_to_the_prompt_and_matches_fresh_logits() {
    let mut env = env_with_state();
    let fresh = match eval_value(&mut env, "gen_logits(gs)").unwrap() {
        Value::Array(a) => a.data().to_vec(),
        v => panic!("{v:?}"),
    };
    eval_value(&mut env, "gen_append(gs, 5)").unwrap();
    eval_value(&mut env, "gen_append(gs, 6)").unwrap();
    assert_eq!(scalar(&mut env, "gen_stats(gs).tokens"), 5.0);
    // reset -> back to prompt length, and logits match the fresh state.
    assert_eq!(scalar(&mut env, "gen_reset(gs)"), 3.0);
    assert_eq!(scalar(&mut env, "gen_stats(gs).tokens"), 3.0);
    let after = match eval_value(&mut env, "gen_logits(gs)").unwrap() {
        Value::Array(a) => a.data().to_vec(),
        v => panic!("{v:?}"),
    };
    assert_eq!(after, fresh, "reset logits are bit-identical to fresh");
}

#[test]
fn multirow_append_equals_sequential_singles() {
    let mut a = env_with_state();
    let mut b = env_with_state();
    // a: one vector append; b: three single appends.
    a_eval(&mut a, "gen_append(gs, [4, 5, 6])");
    for id in [4, 5, 6] {
        a_eval(&mut b, &format!("gen_append(gs, {id})"));
    }
    let la = logits(&mut a);
    let lb = logits(&mut b);
    assert_eq!(la, lb, "vector append == sequential singles");
    assert_eq!(scalar(&mut a, "gen_stats(gs).tokens"), 6.0);
}

fn a_eval(env: &mut Environment, src: &str) {
    eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}"));
}
fn logits(env: &mut Environment) -> Vec<f64> {
    match eval_value(env, "gen_logits(gs)").unwrap() {
        Value::Array(a) => a.data().to_vec(),
        v => panic!("{v:?}"),
    }
}

#[test]
fn controls_error_on_unknown_state() {
    let mut env = Environment::new();
    assert!(eval_value(&mut env, "gen_stats(nope)").is_err());
    assert!(eval_value(&mut env, "gen_reset(nope)").is_err());
    assert!(eval_value(&mut env, "gen_clone(nope)").is_err());
}
