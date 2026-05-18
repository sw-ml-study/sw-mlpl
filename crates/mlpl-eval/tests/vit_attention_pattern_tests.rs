//! `demos/vit_attention_pattern.mlpl` integration test.
//! Saga 29 step 006.
//!
//! Validates the success criteria from docs/milestone-vit.md:
//!
//! - Row sums of the rendered attention matrix are within
//!   1e-6 of 1.0.
//! - The SVG renders to a non-empty string.
//! - The same seed produces the same matrix (determinism).

use std::fs;
use std::path::PathBuf;

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn demo_source() -> String {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../demos/vit_attention_pattern.mlpl");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Run the demo and return the final value (the row-sum
/// vector that the demo emits last) plus the final value of
/// the `A` (attention matrix) binding via a second
/// evaluation that ends in `A`.
fn run_to_row_sums() -> DenseArray {
    let src = demo_source();
    let tokens = lex(&src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).expect("eval");
    match v {
        Value::Array(a) => a,
        other => panic!("expected the row-sum vector to be a Value::Array, got {other:?}"),
    }
}

fn run_to_attention_matrix() -> DenseArray {
    let src = demo_source();
    // The demo's last binding for the attention matrix is
    // `A`; evaluate the demo then read it back.
    let mut full = src;
    full.push_str("\nA\n");
    let tokens = lex(&full).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).expect("eval");
    match v {
        Value::Array(a) => a,
        other => panic!("expected A to be a Value::Array, got {other:?}"),
    }
}

#[test]
fn demo_parses_and_evaluates_without_error() {
    let v = run_to_row_sums();
    assert_eq!(v.shape().dims(), &[17], "expected row sums of length 17");
}

#[test]
fn attention_matrix_has_shape_17_by_17() {
    let a = run_to_attention_matrix();
    assert_eq!(a.shape().dims(), &[17, 17]);
}

#[test]
fn every_row_of_attention_matrix_sums_to_one() {
    let a = run_to_attention_matrix();
    assert_eq!(a.shape().dims(), &[17, 17]);
    for i in 0..17 {
        let row_sum: f64 = a.data()[i * 17..(i + 1) * 17].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "row {i}: sum {row_sum} is not within 1e-6 of 1.0"
        );
    }
}

#[test]
fn demo_row_sums_output_is_all_ones() {
    let v = run_to_row_sums();
    for (i, &s) in v.data().iter().enumerate() {
        assert!(
            (s - 1.0).abs() < 1e-6,
            "row_sums[{i}] = {s} is not within 1e-6 of 1.0"
        );
    }
}

#[test]
fn same_seed_produces_same_attention_matrix() {
    // The demo file embeds seeds (101, 201, 301, 401, 501,
    // 502) so two independent evaluations must produce
    // bitwise-identical matrices.
    let a1 = run_to_attention_matrix();
    let a2 = run_to_attention_matrix();
    assert_eq!(a1.shape().dims(), a2.shape().dims());
    assert_eq!(
        a1.data(),
        a2.data(),
        "two evaluations of the demo produced different matrices"
    );
}

#[test]
fn svg_render_returns_a_non_empty_string() {
    // Re-run a trimmed prefix of the demo + a final
    // `svg(A, "heatmap")` so we can grab the SVG string
    // directly via the final REPL value.
    let mut src = demo_source();
    // Append a final expression that returns the SVG (the
    // demo already calls `svg(...)` for its side-effect
    // print but the final value of the demo is the row
    // sums).
    src.push_str("\nsvg(A, \"heatmap\")\n");
    let tokens = lex(&src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).expect("eval");
    match v {
        Value::Str(s) => {
            assert!(!s.is_empty(), "svg returned an empty string");
            assert!(
                s.contains("<svg"),
                "svg output should contain an <svg tag, got first 80 chars: {}",
                &s[..s.len().min(80)]
            );
        }
        other => panic!("expected Value::Str from svg(), got {other:?}"),
    }
}
