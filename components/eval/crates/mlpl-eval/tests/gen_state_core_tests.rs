//! gen_state / gen_logits / gen_append core: cached generation
//! is BIT-IDENTICAL to full recompute on CPU (the exit criterion
//! in docs/kv-cache-design.md).

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn vec_of(env: &mut Environment, src: &str) -> Vec<f64> {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("expected array from {src}, got {other:?}"),
    }
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Greedy-generate `steps` tokens BOTH ways from the same
/// prompt, asserting the full logits row is bit-identical at
/// every step, and return the generated ids.
fn assert_equivalent(env: &mut Environment, prompt: &[usize], steps: usize) -> Vec<usize> {
    let prompt_src = prompt
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    eval_value(env, &format!("gs = gen_state(m, [{prompt_src}])")).unwrap();
    let mut seq: Vec<usize> = prompt.to_vec();
    let mut generated = Vec::new();
    for step in 0..steps {
        let cached = vec_of(env, "gen_logits(gs)");
        let seq_src = seq
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let full = vec_of(env, &format!("last_row(apply(m, [{seq_src}]))"));
        assert_eq!(
            cached, full,
            "step {step}: cached logits must be bit-identical to recompute"
        );
        let nxt = argmax(&cached);
        eval_value(env, &format!("gen_append(gs, {nxt})")).unwrap();
        seq.push(nxt);
        generated.push(nxt);
    }
    generated
}

#[test]
fn tiny_lm_chain_cached_equals_recompute_bitwise() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "m = chain(embed(16, 8, 0), \
                   residual(chain(rms_norm(8), causal_attention(8, 2, 1))), \
                   rms_norm(8), \
                   linear(8, 16, 4))",
    )
    .unwrap();
    let ids = assert_equivalent(&mut env, &[0, 1, 2], 10);
    assert_eq!(ids.len(), 10);
}

#[test]
fn two_attention_blocks_keep_their_caches_straight() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "m = chain(embed(12, 8, 0), \
                   residual(chain(rms_norm(8), causal_attention(8, 2, 1))), \
                   residual(chain(rms_norm(8), causal_attention(8, 1, 7))), \
                   rms_norm(8), \
                   linear(8, 12, 4))",
    )
    .unwrap();
    assert_equivalent(&mut env, &[3, 1], 6);
}

#[test]
fn engram_in_chain_is_supported() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "m = chain(embed(16, 8, 0), \
                   engram(8, 2, 2, 32, 4, 3), \
                   residual(chain(rms_norm(8), causal_attention(8, 2, 1))), \
                   rms_norm(8), \
                   linear(8, 16, 4))",
    )
    .unwrap();
    assert_equivalent(&mut env, &[0, 5], 4);
}

#[test]
fn non_causal_attention_is_a_tutoring_error() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "m = chain(embed(16, 8, 0), attention(8, 2, 1), linear(8, 16, 4))",
    )
    .unwrap();
    let e = eval_value(&mut env, "gs = gen_state(m, [0, 1])").unwrap_err();
    assert!(
        e.contains("causal_attention") && e.contains("cannot be exact"),
        "tutoring error expected, got: {e}"
    );
}

#[test]
fn misuse_errors_are_structured() {
    let mut env = Environment::new();
    eval_value(&mut env, "m = chain(embed(8, 4, 0), linear(4, 8, 1))").unwrap();
    let e = eval_value(&mut env, "gen_logits(nope)").unwrap_err();
    assert!(e.contains("unknown generation state"), "{e}");
    let e = eval_value(&mut env, "gen_state(chain(linear(4, 4, 1)), [0])").unwrap_err();
    assert!(e.contains("bound to a name"), "{e}");
    let e = eval_value(&mut env, "gen_append(gs2, 1)").unwrap_err();
    assert!(e.contains("unknown generation state"), "{e}");
    // gen_append returns the running token count.
    eval_value(&mut env, "gs = gen_state(m, [0, 1])").unwrap();
    let count = vec_of(&mut env, "gen_append(gs, 3)");
    assert_eq!(count, vec![3.0]);
}
