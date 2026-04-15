//! Saga 13 step 002: sinusoidal positional encoding.
//!
//! `sinusoidal_encoding(seq_len, d_model)` returns a deterministic
//! `[seq_len, d_model]` float array labeled `[time, dim]` using the
//! standard transformer formula:
//!
//! ```text
//! PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
//! PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
//! ```
//!
//! It is a pure function (no parameters, no seed) so two runs with
//! the same args produce identical output. Adding it to the output of
//! a token embedding works element-wise when shapes match; the
//! labeled side propagates `[time, dim]` through the binop.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

#[test]
fn sinusoidal_encoding_has_shape_seq_by_d() {
    let mut env = Environment::new();
    let out = run("sinusoidal_encoding(8, 4)", &mut env);
    assert_eq!(out.shape().dims(), &[8, 4]);
}

#[test]
fn sinusoidal_encoding_position_zero_is_alternating_zero_one() {
    // At pos=0: sin(0)=0, cos(0)=1 for every (i, parity) pair.
    let mut env = Environment::new();
    let out = run("sinusoidal_encoding(8, 4)", &mut env);
    let row0 = &out.data()[0..4];
    assert!((row0[0] - 0.0).abs() < 1e-12, "PE[0,0] = sin(0) = 0");
    assert!((row0[1] - 1.0).abs() < 1e-12, "PE[0,1] = cos(0) = 1");
    assert!((row0[2] - 0.0).abs() < 1e-12, "PE[0,2] = sin(0) = 0");
    assert!((row0[3] - 1.0).abs() < 1e-12, "PE[0,3] = cos(0) = 1");
}

#[test]
fn sinusoidal_encoding_position_one_matches_reference() {
    // For d_model=4, i in {0,1}:
    //   col 0 (i=0, even): sin(pos / 10000^0)         = sin(pos)
    //   col 1 (i=0, odd):  cos(pos / 10000^0)         = cos(pos)
    //   col 2 (i=1, even): sin(pos / 10000^(2/4))     = sin(pos / 100)
    //   col 3 (i=1, odd):  cos(pos / 10000^(2/4))     = cos(pos / 100)
    let mut env = Environment::new();
    let out = run("sinusoidal_encoding(8, 4)", &mut env);
    let row1 = &out.data()[4..8];
    let expected = [
        1.0_f64.sin(),
        1.0_f64.cos(),
        (1.0_f64 / 100.0).sin(),
        (1.0_f64 / 100.0).cos(),
    ];
    for (i, (&got, &want)) in row1.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "PE[1, {i}] = {got}, expected {want}"
        );
    }
}

#[test]
fn sinusoidal_encoding_is_deterministic() {
    // Pure function: same args, same output across runs.
    let mut env = Environment::new();
    let a = run("sinusoidal_encoding(16, 8)", &mut env);
    let b = run("sinusoidal_encoding(16, 8)", &mut env);
    assert_eq!(a.shape(), b.shape());
    for (x, y) in a.data().iter().zip(b.data().iter()) {
        assert!((x - y).abs() < 1e-15, "determinism: {x} vs {y}");
    }
}

#[test]
fn sinusoidal_encoding_carries_time_dim_labels() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("PE = sinusoidal_encoding(4, 6)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let out = env.get("PE").expect("PE bound");
    let labels = out
        .labels()
        .expect("sinusoidal_encoding output must be labeled");
    assert_eq!(labels.len(), 2);
    assert_eq!(labels[0].as_deref(), Some("time"));
    assert_eq!(labels[1].as_deref(), Some("dim"));
}

#[test]
fn embed_plus_sinusoidal_encoding_propagates_labels() {
    // Pattern documented for adding positional info to a token
    // embedding output. With toks of shape [T], `apply(E, toks)`
    // returns an unlabeled [T, d_model] array. The sinusoidal table
    // carries `[time, dim]` labels; same-shape elementwise add lets
    // those labels propagate to the result. For batched [B, T, D]
    // workflows the same primitive is applied after flattening
    // tokens to [B*T] and tiling positions accordingly.
    let mut env = Environment::new();
    eval_program(
        &parse(
            &lex("E = embed(5, 4, 7)\n\
                  toks = [0.0, 1.0, 2.0, 3.0]\n\
                  emb = apply(E, toks)\n\
                  PE = sinusoidal_encoding(4, 4)\n\
                  pos_emb = emb + PE")
            .unwrap(),
        )
        .unwrap(),
        &mut env,
    )
    .unwrap();

    let out = env.get("pos_emb").expect("pos_emb bound");
    assert_eq!(out.shape().dims(), &[4, 4]);
    let labels = out.labels().expect("labels propagate from PE");
    assert_eq!(labels[0].as_deref(), Some("time"));
    assert_eq!(labels[1].as_deref(), Some("dim"));

    // Sanity: the result should equal emb + PE element-wise. Pull
    // the underlying arrays back out and verify a single position.
    let emb = env.get("emb").unwrap();
    let pe = env.get("PE").unwrap();
    for i in 0..emb.data().len() {
        let want = emb.data()[i] + pe.data()[i];
        let got = out.data()[i];
        assert!((got - want).abs() < 1e-12, "elem {i}: {got} vs {want}");
    }
}

#[test]
fn sinusoidal_encoding_rejects_non_scalar_args() {
    let mut env = Environment::new();
    env.set(
        "sl".into(),
        DenseArray::new(Shape::new(vec![2]), vec![4.0, 4.0]).unwrap(),
    );
    let result = eval_program(
        &parse(&lex("sinusoidal_encoding(sl, 4)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "non-scalar seq_len must error");
}

#[test]
fn sinusoidal_encoding_rejects_zero_d_model() {
    let mut env = Environment::new();
    let result = eval_program(
        &parse(&lex("sinusoidal_encoding(4, 0)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(result.is_err(), "d_model=0 must error");
}
