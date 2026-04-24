//! Saga 16 step 004: end-to-end `demos/embedding_viz.mlpl`
//! integration test.
//!
//! Cut-down variant (V=9, d=4, 3 train steps, 3 cluster
//! centers in 4-D) exercises the pipeline in under a
//! second on CPU. Pins:
//!
//! - Base training produces finite embedding values.
//! - `tsne(table, ...)` returns shape `[V, 2]` with
//!   finite values.
//! - `knn(table, k)` returns shape `[V, k]` with every
//!   index in `[0, V)` and row `i` excluded from its own
//!   neighbor list.
//! - A column-selector matmul produces `[V, 3]` for the
//!   3-D scatter input.

use mlpl_eval::{Environment, eval_program, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap();
}

fn run_string(env: &mut Environment, src: &str) -> String {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    match eval_program_value(&stmts, env).unwrap() {
        mlpl_eval::Value::Str(s) => s,
        other => panic!("expected Value::Str, got {other:?}"),
    }
}

/// Shared setup: build a 3-cluster target in 4-D, train a
/// standalone `embed(V, d, 0)` toward it via MSE for 3
/// adam steps, extract the learned `table`.
const BASE_SETUP: &str = "\
V = 9 ; d = 4\n\
cluster_assign = reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                          0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, \
                          0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], \
                         [9, 3])\n\
centers = reshape([2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0], [3, 4])\n\
target = matmul(cluster_assign, centers) + randn(1, [9, 4]) * 0.1\n\
emb = embed(V, d, 0)\n\
ids = iota(V)\n\
train 3 { \
  adam(mean((apply(emb, ids) - target) * (apply(emb, ids) - target)), emb, \
       0.1, 0.9, 0.999, 0.00000001); \
  loss_metric = mean((apply(emb, ids) - target) * (apply(emb, ids) - target)) \
}\n\
table = apply(emb, iota(V))\n\
";

#[test]
fn embedding_viz_tsne_shape_and_finiteness() {
    let mut env = Environment::new();
    run(&mut env, BASE_SETUP);
    run(&mut env, "emb_2d = tsne(table, 2.0, 100, 7)");
    let emb_2d = env.get("emb_2d").unwrap();
    assert_eq!(emb_2d.shape().dims(), &[9, 2]);
    for v in emb_2d.data() {
        assert!(v.is_finite(), "tsne output element {v} should be finite");
    }
}

#[test]
fn embedding_viz_knn_self_excluded_and_in_range() {
    let mut env = Environment::new();
    run(&mut env, BASE_SETUP);
    run(&mut env, "neighbors = knn(table, 3)");
    let neighbors = env.get("neighbors").unwrap();
    assert_eq!(neighbors.shape().dims(), &[9, 3]);
    for i in 0..9_usize {
        for j in 0..3_usize {
            let idx = neighbors.data()[i * 3 + j] as usize;
            assert!(idx < 9, "row {i}: neighbor index {idx} out of range");
            assert_ne!(idx, i, "row {i}: self-index must not appear in neighbors");
        }
    }
}

#[test]
fn embedding_viz_3d_projection_shape() {
    let mut env = Environment::new();
    run(&mut env, BASE_SETUP);
    // Column selector: pick first 3 of the d=4 dims.
    run(
        &mut env,
        "selector = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [4, 3])",
    );
    run(&mut env, "emb_3d = matmul(table, selector)");
    let emb_3d = env.get("emb_3d").unwrap();
    assert_eq!(emb_3d.shape().dims(), &[9, 3]);
    for v in emb_3d.data() {
        assert!(v.is_finite(), "3-D projection element {v} should be finite");
    }
}

#[test]
fn embedding_viz_svg_renders_without_panic() {
    let mut env = Environment::new();
    run(&mut env, BASE_SETUP);
    run(&mut env, "emb_2d = tsne(table, 2.0, 100, 7)");
    run(
        &mut env,
        "selector = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [4, 3])",
    );
    run(&mut env, "emb_3d = matmul(table, selector)");

    let scatter_svg = run_string(&mut env, "svg(emb_2d, \"scatter\")");
    assert!(scatter_svg.starts_with("<svg"));
    assert!(scatter_svg.ends_with("</svg>"));

    let scatter3d_svg = run_string(&mut env, "svg(emb_3d, \"scatter3d\")");
    assert!(scatter3d_svg.starts_with("<svg"));
    assert!(scatter3d_svg.ends_with("</svg>"));
    // Scatter3d always renders axis gizmos with X/Y/Z labels.
    for label in ["X", "Y", "Z"] {
        assert!(
            scatter3d_svg.contains(&format!(">{label}<")),
            "scatter3d should include axis label {label}"
        );
    }
}
