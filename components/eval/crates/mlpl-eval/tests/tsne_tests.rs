//! Saga 16 step 002: `tsne(X, perplexity, iters, seed)`.
//!
//! Classic van der Maaten t-SNE: per-row perplexity-
//! calibrated conditional probabilities in the high-dim
//! space, Student's-t affinities in the low-dim space, KL
//! divergence loss, gradient descent. Output is rank-2
//! `[N, 2]`.
//!
//! Tests pin three claims:
//! 1. Structure is preserved: well-separated clusters in
//!    the input stay well-separated in the output.
//! 2. Shape: any `[N, D]` input gives a `[N, 2]` output.
//! 3. Determinism: identical `(X, perplexity, iters,
//!    seed)` produces bit-identical output.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run_expr(env: &mut Environment, src: &str) -> DenseArray {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap()
}

/// Build three well-separated 4-D Gaussian clusters of 20
/// points each. Returns a `[60, 4]` array whose first 20
/// rows are cluster 0, next 20 are cluster 1, last 20 are
/// cluster 2.
fn three_cluster_fixture() -> DenseArray {
    let centers = [
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
    ];
    // Small deterministic perturbations so points within a
    // cluster are not identical (t-SNE cannot compute a
    // binary search for perplexity on identical points).
    let mut data = Vec::with_capacity(60 * 4);
    for (c_idx, center) in centers.iter().enumerate() {
        for p in 0..20 {
            for (d, coord) in center.iter().enumerate() {
                let jitter = ((c_idx * 20 + p) as f64 * 0.017 + d as f64 * 0.031).sin() * 0.3;
                data.push(coord + jitter);
            }
        }
    }
    arr(vec![60, 4], data)
}

/// Centroid of rows `start..end` (inclusive, exclusive) in
/// a `[N, 2]` t-SNE output.
fn centroid_2d(y: &DenseArray, start: usize, end: usize) -> [f64; 2] {
    let n = (end - start) as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in start..end {
        cx += y.data()[i * 2];
        cy += y.data()[i * 2 + 1];
    }
    [cx / n, cy / n]
}

fn dist2_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

#[test]
fn tsne_preserves_three_cluster_structure() {
    let mut env = Environment::new();
    env.set("X".into(), three_cluster_fixture());
    let y = run_expr(&mut env, "tsne(X, 15.0, 250, 42)");
    assert_eq!(y.shape().dims(), &[60, 2]);

    // Cluster centroids (rows 0..20, 20..40, 40..60).
    let c0 = centroid_2d(&y, 0, 20);
    let c1 = centroid_2d(&y, 20, 40);
    let c2 = centroid_2d(&y, 40, 60);

    // Compute intra-cluster spread (max distance from the
    // centroid) vs the inter-cluster centroid separation.
    // If t-SNE preserved the input structure, inter >>
    // intra.
    let max_intra = (0..60)
        .map(|i| {
            let (c, _) = match i {
                0..=19 => (c0, 0),
                20..=39 => (c1, 1),
                _ => (c2, 2),
            };
            dist2_2d([y.data()[i * 2], y.data()[i * 2 + 1]], c).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let inter = [
        dist2_2d(c0, c1).sqrt(),
        dist2_2d(c0, c2).sqrt(),
        dist2_2d(c1, c2).sqrt(),
    ];
    let min_inter = inter.iter().copied().fold(f64::INFINITY, f64::min);

    // Loose but meaningful bound: inter-cluster separation
    // should exceed intra-cluster spread by at least 2x.
    assert!(
        min_inter > 2.0 * max_intra,
        "t-SNE output did not preserve 3-cluster structure: \
         min_inter={min_inter}, max_intra={max_intra}, \
         centroids=[{c0:?}, {c1:?}, {c2:?}]"
    );
}

#[test]
fn tsne_output_shape_is_n_by_2() {
    let mut env = Environment::new();
    // 12 points in 5-D; small perplexity so binary search
    // fits in a 5-D space at N=12.
    env.set(
        "X".into(),
        arr(
            vec![12, 5],
            (0..60).map(|i| (i as f64 * 0.1).sin()).collect(),
        ),
    );
    let y = run_expr(&mut env, "tsne(X, 3.0, 50, 1)");
    assert_eq!(y.shape().dims(), &[12, 2]);
    for v in y.data() {
        assert!(v.is_finite(), "tsne output should be finite, got {v}");
    }
}

#[test]
fn tsne_is_deterministic_under_identical_inputs() {
    let mut env = Environment::new();
    env.set("X".into(), three_cluster_fixture());
    let y1 = run_expr(&mut env, "tsne(X, 15.0, 100, 7)");
    let y2 = run_expr(&mut env, "tsne(X, 15.0, 100, 7)");
    assert_eq!(y1.shape().dims(), y2.shape().dims());
    assert_eq!(
        y1.data(),
        y2.data(),
        "tsne output should be bit-identical under identical inputs"
    );
}

#[test]
fn tsne_different_seeds_produce_different_outputs() {
    let mut env = Environment::new();
    env.set("X".into(), three_cluster_fixture());
    let y1 = run_expr(&mut env, "tsne(X, 15.0, 100, 1)");
    let y2 = run_expr(&mut env, "tsne(X, 15.0, 100, 2)");
    assert_ne!(
        y1.data(),
        y2.data(),
        "different seeds should produce different embeddings"
    );
}

#[test]
fn tsne_rejects_non_rank2_input() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![6], (0..6).map(|i| i as f64).collect()));
    let stmts = parse(&lex("tsne(v, 2.0, 10, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("rank-1 X should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("tsne") && (msg.contains("rank") || msg.contains("vector")),
        "got: {msg}"
    );
}

#[test]
fn tsne_rejects_non_positive_perplexity() {
    let mut env = Environment::new();
    env.set("X".into(), three_cluster_fixture());
    let stmts = parse(&lex("tsne(X, 0.0, 10, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("perplexity=0 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("tsne") && msg.contains("perplexity"),
        "got: {msg}"
    );
}

#[test]
fn tsne_rejects_perplexity_equal_or_greater_than_n() {
    let mut env = Environment::new();
    // 8 points; perplexity must be < N.
    env.set(
        "X".into(),
        arr(vec![8, 3], (0..24).map(|i| i as f64 * 0.1).collect()),
    );
    let stmts = parse(&lex("tsne(X, 8.0, 10, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("perplexity=N should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("tsne") && msg.contains("perplexity"),
        "got: {msg}"
    );
}

#[test]
fn tsne_rejects_non_positive_iters() {
    let mut env = Environment::new();
    env.set("X".into(), three_cluster_fixture());
    let stmts = parse(&lex("tsne(X, 15.0, 0, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("iters=0 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("tsne") && msg.contains("iters"), "got: {msg}");
}

#[test]
fn tsne_rejects_wrong_arity() {
    let mut env = Environment::new();
    env.set("X".into(), three_cluster_fixture());
    let stmts = parse(&lex("tsne(X, 15.0, 100)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("3-arg form should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("tsne") || msg.contains("arity"), "got: {msg}");
}
