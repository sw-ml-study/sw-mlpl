//! `engram_stats` builtin (saga E3): addressing statistics
//! (lookups, unique rows, collisions) under the frozen hash
//! contract, memory-table health, and optional gate activation
//! stats -- returned as a record with addressable fields.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> Result<mlpl_array::DenseArray, mlpl_eval::EvalError> {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env)
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    eval(env, src).unwrap().data()[0]
}

#[test]
fn addressing_counts_match_the_hash_contract() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(4, [2], 1, 8, 4, 7); ids = [1, 2, 3]").unwrap();
    eval(&mut env, "s = engram_stats(e, ids); 0").unwrap();
    // 3 positions x 1 order x 1 head.
    assert_eq!(scalar(&mut env, "s.rows_addressed"), 3.0);
    // unique_rows must agree with a direct ngram_hash of the same ids
    // (order-2 bigrams (PAD,1), (1,2), (2,3) are all distinct
    // contexts, so any row sharing is a collision).
    let hashes = eval(&mut env, "ngram_hash(ids, [2], 1, 8, 7)").unwrap();
    let unique: std::collections::HashSet<u64> = hashes.data().iter().map(|&v| v as u64).collect();
    assert_eq!(
        scalar(&mut env, "s.unique_rows"),
        unique.len() as f64,
        "unique_rows must match the frozen ngram_hash contract"
    );
    assert_eq!(
        scalar(&mut env, "s.collisions"),
        3.0 - unique.len() as f64,
        "3 distinct contexts minus distinct rows"
    );
}

#[test]
fn tiny_slot_count_forces_collisions() {
    let mut env = Environment::new();
    // slots=2: five distinct bigram contexts cannot fit 2 rows.
    eval(
        &mut env,
        "e = engram(4, [2], 1, 2, 4, 11); ids = [1, 2, 3, 4, 5]; s = engram_stats(e, ids); 0",
    )
    .unwrap();
    let unique = scalar(&mut env, "s.unique_rows");
    assert!((1.0..=2.0).contains(&unique));
    assert_eq!(
        scalar(&mut env, "s.collisions"),
        5.0 - unique,
        "5 distinct contexts squeezed into {unique} rows"
    );
}

#[test]
fn repeated_context_is_not_a_collision() {
    let mut env = Environment::new();
    // ids [5, 5, 5]: positions 1 and 2 share the SAME bigram (5,5),
    // which is repetition, not collision. 64 slots keeps the two
    // distinct contexts ((PAD,5) and (5,5)) from actually colliding.
    eval(
        &mut env,
        "e = engram(4, [2], 1, 64, 4, 7); s = engram_stats(e, [5, 5, 5]); 0",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "s.rows_addressed"), 3.0);
    assert_eq!(scalar(&mut env, "s.unique_rows"), 2.0);
    assert_eq!(scalar(&mut env, "s.collisions"), 0.0);
}

#[test]
fn memory_health_tracks_the_table() {
    let mut env = Environment::new();
    eval(&mut env, "e = engram(4, [2], 1, 8, 4, 7)").unwrap();
    // Fresh engram: zero memory.
    eval(&mut env, "s0 = engram_stats(e, [1, 2, 3]); 0").unwrap();
    assert_eq!(scalar(&mut env, "s0.nonzero_rows"), 0.0);
    assert_eq!(scalar(&mut env, "s0.max_row_norm"), 0.0);
    // Poke one row to [3, 4, 0, 0]: norm 5.
    let mem_name = env.get_model("e").unwrap().params()[0].clone();
    let mem = env.get(&mem_name).unwrap().clone();
    let mut data = mem.data().to_vec();
    data[2 * 4] = 3.0;
    data[2 * 4 + 1] = 4.0;
    env.set(
        mem_name,
        mlpl_array::DenseArray::new(mem.shape().clone(), data).unwrap(),
    );
    eval(&mut env, "s1 = engram_stats(e, [1, 2, 3]); 0").unwrap();
    assert_eq!(scalar(&mut env, "s1.nonzero_rows"), 1.0);
    assert!((scalar(&mut env, "s1.max_row_norm") - 5.0).abs() < 1e-12);
}

#[test]
fn gate_stats_on_a_fresh_engram_pin_sigmoid_of_the_bias() {
    let mut env = Environment::new();
    // h = zeros and zero memory => gate pre-activation is exactly
    // the -2 bias, so mean == max == sigmoid(-2).
    eval(
        &mut env,
        "e = engram(4, [2], 1, 8, 4, 7); h = reshape(zeros([12]), [3, 4]); \
         s = engram_stats(e, h, [1, 2, 3]); 0",
    )
    .unwrap();
    let expected = 1.0 / (1.0 + (2.0_f64).exp());
    assert!((scalar(&mut env, "s.gate_mean") - expected).abs() < 1e-12);
    assert!((scalar(&mut env, "s.gate_max") - expected).abs() < 1e-12);
    // The two-argument form must NOT carry gate fields.
    eval(&mut env, "s2 = engram_stats(e, [1, 2, 3]); 0").unwrap();
    assert!(eval(&mut env, "s2.gate_mean").is_err());
}

#[test]
fn error_surface_is_loud() {
    let mut env = Environment::new();
    eval(
        &mut env,
        "e = engram(4, [2], 1, 8, 4, 7); m = linear(4, 4, 0)",
    )
    .unwrap();
    let e1 = eval(&mut env, "engram_stats(m, [1, 2])").unwrap_err();
    assert!(format!("{e1}").contains("not an engram"), "{e1}");
    let e2 = eval(&mut env, "engram_stats(e, [1, 2.5])").unwrap_err();
    assert!(format!("{e2}").contains("integer"), "{e2}");
    let e3 = eval(&mut env, "engram_stats(e)").unwrap_err();
    assert!(format!("{e3}").contains("engram_stats"), "{e3}");
}
