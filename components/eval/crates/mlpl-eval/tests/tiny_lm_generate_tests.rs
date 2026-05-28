//! Saga 13 step 007: generation loop + attention-map visualization.
//!
//! Tests cover:
//! 1. `concat(a, b)` -- 1-D array concatenation.
//! 2. `last_row(matrix)` -- extract the last row of a rank-2 matrix.
//! 3. Deterministic generation: same seed produces identical output.
//! 4. `attention_weights(model, X)` -- causal weights are
//!    lower-triangular (zeros above the diagonal).

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

// ---- concat ----

#[test]
fn concat_two_vectors() {
    let mut env = Environment::new();
    let out = run("concat([1, 2, 3], [4, 5])", &mut env);
    assert_eq!(out.shape().dims(), &[5]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn concat_vector_and_scalar() {
    let mut env = Environment::new();
    let out = run("concat([1, 2, 3], 4)", &mut env);
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn concat_scalar_and_vector() {
    let mut env = Environment::new();
    let out = run("concat(0, [1, 2, 3])", &mut env);
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.data(), &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn concat_two_scalars() {
    let mut env = Environment::new();
    let out = run("concat(10, 20)", &mut env);
    assert_eq!(out.shape().dims(), &[2]);
    assert_eq!(out.data(), &[10.0, 20.0]);
}

#[test]
fn concat_rejects_rank2() {
    let mut env = Environment::new();
    let result = eval_program(
        &parse(&lex("concat(reshape(iota(4), [2, 2]), [1])").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "rank-2 concat must error");
}

// ---- last_row ----

#[test]
fn last_row_of_rank2() {
    let mut env = Environment::new();
    let out = run("last_row(reshape(iota(6), [2, 3]))", &mut env);
    assert_eq!(out.shape().dims(), &[3]);
    assert_eq!(out.data(), &[3.0, 4.0, 5.0]);
}

#[test]
fn last_row_of_single_row() {
    let mut env = Environment::new();
    let out = run("last_row(reshape(iota(4), [1, 4]))", &mut env);
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.data(), &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn last_row_rejects_rank1() {
    let mut env = Environment::new();
    let result = eval_program(
        &parse(&lex("last_row([1, 2, 3])").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "rank-1 must error");
}

// ---- generation determinism ----

#[test]
fn generation_is_deterministic_on_fixed_seed() {
    // Train a tiny model, then generate twice with the same seed.
    // The two outputs must be byte-identical.
    let setup = "\
        corpus = \"abcabcabcabcabcabcabcabcabcabcabcabc\"\n\
        ids    = tokenize_bytes(corpus)\n\
        X_all  = shift_pairs_x(ids, 8)\n\
        Y_all  = shift_pairs_y(ids, 8)\n\
        X      = reshape(X_all, [reduce_mul(shape(X_all))])\n\
        Y      = reshape(Y_all, [reduce_mul(shape(Y_all))])\n\
        V      = 256 ; d = 8\n\
        model  = chain(embed(V, d, 0), \
                       residual(chain(rms_norm(d), causal_attention(d, 1, 1))), \
                       rms_norm(d), \
                       linear(d, V, 4))\n\
        train 20 { \
          adam(cross_entropy(apply(model, X), Y), model, 0.05, 0.9, 0.999, 0.00000001); \
          cross_entropy(apply(model, X), Y) \
        }";
    let gen_src = "\
        prompt = tokenize_bytes(\"abc\")\n\
        seq    = prompt\n\
        repeat 10 { \
          logits = apply(model, seq); \
          last   = last_row(logits); \
          nxt    = sample(top_k(last, 40), 0.8, step); \
          seq    = concat(seq, nxt) \
        }\n\
        seq";

    let mut env1 = Environment::new();
    run(setup, &mut env1);
    let out1 = run(gen_src, &mut env1);

    let mut env2 = Environment::new();
    run(setup, &mut env2);
    let out2 = run(gen_src, &mut env2);

    assert_eq!(out1.shape(), out2.shape(), "shapes must match");
    assert_eq!(out1.data(), out2.data(), "data must match byte-for-byte");
    // prompt was 3 tokens, generated 10 more => 13 total
    assert_eq!(out1.shape().dims(), &[13]);
}

// ---- attention_weights ----

#[test]
fn attention_weights_are_lower_triangular_for_causal() {
    let mut env = Environment::new();
    let src = "\
        d = 4 ; h = 1\n\
        model = causal_attention(d, h, 42)\n\
        X = randn(7, [5, 4])\n\
        attention_weights(model, X)";
    let weights = run(src, &mut env);
    // Single head => [T, T] = [5, 5]
    assert_eq!(weights.shape().dims(), &[5, 5]);
    let t = 5;
    for r in 0..t {
        for c in 0..t {
            let v = weights.data()[r * t + c];
            if c > r {
                // Above diagonal must be zero (or near-zero after softmax
                // of -1e9)
                assert!(
                    v.abs() < 1e-6,
                    "weight[{r},{c}] = {v} should be ~0 above diagonal"
                );
            }
        }
        // Each row must sum to ~1 (softmax output)
        let row_sum: f64 = (0..t).map(|c| weights.data()[r * t + c]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "row {r} sum = {row_sum}, expected ~1.0"
        );
    }
}

#[test]
fn attention_weights_non_causal_has_more_above_diagonal_than_causal() {
    // Non-causal must have strictly more above-diagonal weight than
    // causal (which masks those entries to ~0).
    let mut env = Environment::new();
    run("d = 4 ; h = 1", &mut env);
    run("X = randn(7, [5, 4])", &mut env);
    run("causal_model = causal_attention(d, h, 42)", &mut env);
    run("noncausal_model = attention(d, h, 42)", &mut env);
    let causal_w = run("attention_weights(causal_model, X)", &mut env);
    let noncausal_w = run("attention_weights(noncausal_model, X)", &mut env);
    let t = 5;
    let above_causal: f64 = (0..t)
        .flat_map(|r| (r + 1..t).map(move |c| (r, c)))
        .map(|(r, c)| causal_w.data()[r * t + c])
        .sum();
    let above_noncausal: f64 = (0..t)
        .flat_map(|r| (r + 1..t).map(move |c| (r, c)))
        .map(|(r, c)| noncausal_w.data()[r * t + c])
        .sum();
    assert!(
        above_noncausal > above_causal + 1e-4,
        "non-causal above-diagonal sum ({above_noncausal}) \
         must exceed causal ({above_causal})"
    );
}
