//! `gather_rows` on the tape is a native row gather: forward copies the
//! addressed rows, backward scatter-ADDS into only those rows (duplicate
//! ids accumulate) -- O(n * d), with no `[n, vocab]` one-hot selection
//! matrix. Its gradient must equal the dense one-hot formulation exactly,
//! and a large-vocabulary gather must be fast. The `embed` layer's tape
//! lookup uses the same node.

use std::time::{Duration, Instant};

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Value, EvalError> {
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        last = eval_program_value(&parse(&lex(line).unwrap()).unwrap(), env)?;
    }
    Ok(last)
}

fn nums(env: &mut Environment, src: &str) -> Vec<f64> {
    match run(env, src) {
        Ok(Value::Array(a)) => a.data().to_vec(),
        other => panic!("{src}: {other:?}"),
    }
}

#[test]
fn gradient_matches_the_dense_one_hot_form_with_duplicates() {
    let mut env = Environment::new();
    run(
        &mut env,
        "T = param[5, 3]\nT = randn(1, [5, 3])\nids = [4, 1, 4, 0, 4]\nw = randn(2, [5, 3])",
    )
    .unwrap();
    let native = nums(&mut env, "grad(reduce_add(gather_rows(T, ids) * w), T)");
    let dense = nums(
        &mut env,
        "grad(reduce_add(matmul(one_hot(ids, 5), T) * w), T)",
    );
    assert_eq!(native.len(), 15);
    for (a, b) in native.iter().zip(&dense) {
        assert!((a - b).abs() < 1e-12, "{native:?} vs {dense:?}");
    }
    // Row 4 is addressed three times: its gradient accumulates all three.
    assert!(native[12..15].iter().any(|v| *v != 0.0));
    // Rows 2 and 3 are never addressed: zero gradient.
    assert!(native[6..12].iter().all(|v| *v == 0.0));
}

#[test]
fn rank2_indices_keep_their_shape_and_embed_trains() {
    let mut env = Environment::new();
    run(&mut env, "T = param[4, 2]\nT = randn(3, [4, 2])").unwrap();
    assert_eq!(
        nums(&mut env, "shape(gather_rows(T, [[0, 1], [3, 3]]))"),
        vec![2.0, 2.0, 2.0]
    );
    run(
        &mut env,
        "E = embed(6, 3, 1)\nids = [1, 2, 2, 5]\nY = randn(4, [4, 3])\n\
         train 20 { adam(mean((apply(E, ids) - Y) * (apply(E, ids) - Y)), E, 0.05, 0.9, 0.999, 0.00000001) }",
    )
    .unwrap();
    let l = nums(&mut env, "last_losses");
    assert!(l[19] < l[0], "{} -> {}", l[0], l[19]);
}

#[test]
fn a_large_vocabulary_gather_is_fast() {
    let mut env = Environment::new();
    // The dense form would build a 20000 x 50000 one-hot (8 GB) here.
    run(
        &mut env,
        "T = param[50000, 16]\nT = zeros([50000, 16])\nids = range(20000)",
    )
    .unwrap();
    let start = Instant::now();
    let g = nums(&mut env, "grad(reduce_add(gather_rows(T, ids)), T)");
    assert_eq!(g.len(), 50000 * 16);
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
}
