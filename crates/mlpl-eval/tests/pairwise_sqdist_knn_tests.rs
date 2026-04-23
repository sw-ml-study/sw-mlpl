//! Saga 16 step 001: `pairwise_sqdist(X)` + `knn(X, k)`.
//!
//! Two sibling builtins that cover the distance-based
//! inspection surface for an embedding table (or any
//! rank-2 array viewed as a set of points).
//! `pairwise_sqdist` returns `[N, N]` squared Euclidean
//! distances; `knn` returns the `k` nearest non-self
//! neighbors per row, sorted by ascending distance, ties
//! broken by lower original index (stable).

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

// ---- pairwise_sqdist ----

#[test]
fn pairwise_sqdist_three_points_in_2d() {
    // X = [[0,0], [1,0], [0,1]]
    // Pairwise squared distances:
    //   D[0,0]=0  D[0,1]=1  D[0,2]=1
    //   D[1,0]=1  D[1,1]=0  D[1,2]=2
    //   D[2,0]=1  D[2,1]=2  D[2,2]=0
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(vec![3, 2], vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    );
    let d = run_expr(&mut env, "pairwise_sqdist(X)");
    assert_eq!(d.shape().dims(), &[3, 3]);
    assert_eq!(d.data(), &[0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0]);
}

#[test]
fn pairwise_sqdist_has_zero_diagonal() {
    let mut env = Environment::new();
    // Arbitrary 4 points in 3D
    env.set(
        "X".into(),
        arr(
            vec![4, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.0, 2.0, 3.0, -3.0, 1.0],
        ),
    );
    let d = run_expr(&mut env, "pairwise_sqdist(X)");
    for i in 0..4 {
        assert_eq!(d.data()[i * 4 + i], 0.0, "diagonal D[{i},{i}] should be 0");
    }
}

#[test]
fn pairwise_sqdist_is_symmetric() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(
            vec![4, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.0, 2.0, 3.0, -3.0, 1.0],
        ),
    );
    let d = run_expr(&mut env, "pairwise_sqdist(X)");
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (d.data()[i * 4 + j] - d.data()[j * 4 + i]).abs() < 1e-12,
                "D[{i},{j}] != D[{j},{i}]"
            );
        }
    }
}

#[test]
fn pairwise_sqdist_rejects_non_rank2() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![4], vec![0.0, 1.0, 2.0, 3.0]));
    let stmts = parse(&lex("pairwise_sqdist(v)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("rank-1 input should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("pairwise_sqdist") && (msg.contains("rank") || msg.contains("rank-2")),
        "error should reference pairwise_sqdist + rank, got: {msg}"
    );
}

#[test]
fn pairwise_sqdist_handles_empty_input() {
    // N=0 should produce [0, 0] gracefully rather than panic.
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![0, 3], vec![]));
    let d = run_expr(&mut env, "pairwise_sqdist(X)");
    assert_eq!(d.shape().dims(), &[0, 0]);
    assert!(d.data().is_empty());
}

// ---- knn ----

#[test]
fn knn_one_nearest_on_a_line() {
    // Points on a 1-D line at 0, 1, 2, 3.
    //   knn(X, 1)[0] = [1] (only 1, 2, 3; 1 is nearest)
    //   knn(X, 1)[1] = [0] (0 and 2 tie by distance; 0 wins by lower index)
    //   knn(X, 1)[2] = [1] (1 and 3 tie; 1 wins)
    //   knn(X, 1)[3] = [2]
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]));
    let idx = run_expr(&mut env, "knn(X, 1)");
    assert_eq!(idx.shape().dims(), &[4, 1]);
    assert_eq!(idx.data(), &[1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn knn_two_nearest_sorted_by_distance() {
    // knn(X, 2):
    //   Row 0 (point 0): distances to 1,2,3 = 1,4,9 -> [1, 2]
    //   Row 1 (point 1): distances to 0,2,3 = 1,1,4 -> tie at (0,2); lower idx first -> [0, 2]
    //   Row 2 (point 2): distances to 0,1,3 = 4,1,1 -> tie at (1,3); lower idx first -> [1, 3]
    //   Row 3 (point 3): distances to 0,1,2 = 9,4,1 -> [2, 1]
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]));
    let idx = run_expr(&mut env, "knn(X, 2)");
    assert_eq!(idx.shape().dims(), &[4, 2]);
    assert_eq!(idx.data(), &[1.0, 2.0, 0.0, 2.0, 1.0, 3.0, 2.0, 1.0]);
}

#[test]
fn knn_self_excluded() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        arr(
            vec![5, 2],
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        ),
    );
    let idx = run_expr(&mut env, "knn(X, 4)");
    assert_eq!(idx.shape().dims(), &[5, 4]);
    // For every row i, assert i is not present in idx[i].
    for i in 0..5 {
        let row_start = i * 4;
        for j in 0..4 {
            let v = idx.data()[row_start + j] as usize;
            assert_ne!(v, i, "row {i}: self-index appeared at position {j}");
        }
    }
}

#[test]
fn knn_rejects_k_zero() {
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![3, 2], vec![0.0; 6]));
    let stmts = parse(&lex("knn(X, 0)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("k=0 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("knn") && (msg.contains("k") || msg.contains("positive")),
        "got: {msg}"
    );
}

#[test]
fn knn_rejects_k_equal_to_n() {
    // With self-exclusion we can have at most N-1 neighbors.
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![3, 2], vec![0.0; 6]));
    let stmts = parse(&lex("knn(X, 3)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("k=N should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("knn") && msg.contains("k"), "got: {msg}");
}

#[test]
fn knn_rejects_non_rank2_input() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![4], vec![0.0, 1.0, 2.0, 3.0]));
    let stmts = parse(&lex("knn(v, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("rank-1 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("knn") && (msg.contains("rank") || msg.contains("vector")),
        "got: {msg}"
    );
}

#[test]
fn knn_rejects_non_integer_k() {
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![3, 2], vec![0.0; 6]));
    let stmts = parse(&lex("knn(X, 1.5)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("non-integer k should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("knn"), "got: {msg}");
}

#[test]
fn knn_rejects_wrong_arity() {
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![3, 2], vec![0.0; 6]));
    let stmts = parse(&lex("knn(X)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("1-arg form should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("knn") || msg.contains("arity"), "got: {msg}");
}
