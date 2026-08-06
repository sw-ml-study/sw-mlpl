//! transpose_axes(x, perm) -- generalized dyadic transpose
//! (APL2's P transpose A) on flat arrays: result axis i draws
//! from source axis perm[i], 0-based like rotate/compress.

use mlpl_eval::Environment;
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<mlpl_array::DenseArray, String> {
    let stmts = parse(&lex(src).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program(&stmts, env).map_err(|e| e.to_string())
}

#[test]
fn rank2_swap_matches_transpose() {
    let mut env = Environment::new();
    let a = run(
        &mut env,
        "transpose_axes(reshape(range(6), [2, 3]), [1, 0])",
    )
    .unwrap();
    let b = run(&mut env, "transpose(reshape(range(6), [2, 3]))").unwrap();
    assert_eq!(a.shape().dims(), &[3, 2]);
    assert_eq!(a.data(), b.data());
}

#[test]
fn identity_perm_is_a_clone() {
    let mut env = Environment::new();
    let a = run(
        &mut env,
        "transpose_axes(reshape(range(24), [2, 3, 4]), [0, 1, 2])",
    )
    .unwrap();
    assert_eq!(a.shape().dims(), &[2, 3, 4]);
    assert_eq!(a.data()[5], 5.0);
}

#[test]
fn sudoku_block_reranking() {
    // A 9x9 board holding 10*row + col; after reshape to
    // [3,3,3,3] and the APL2 `1 3 2 4` swap ([0,2,1,3] 0-based),
    // slice [0][1] must be the TOP-MIDDLE 3x3 block.
    let mut env = Environment::new();
    run(
        &mut env,
        "board = transpose_axes(reshape(range(81), [3, 3, 3, 3]), [0, 2, 1, 3])",
    )
    .unwrap();
    let blocks = run(&mut env, "board").unwrap();
    assert_eq!(blocks.shape().dims(), &[3, 3, 3, 3]);
    // range(81) as a 9x9 has cell (r, c) = 9r + c. Top-middle
    // block (block-row 0, block-col 1) starts at (0, 3):
    let d = blocks.data();
    // blocks[0][1][i][j] should be 9*i + 3 + j
    for i in 0..3 {
        for j in 0..3 {
            // block-row 0, block-col 1 -> outer offset 1
            let idx = (3 + i) * 3 + j;
            assert_eq!(d[idx], (9 * i + 3 + j) as f64, "block[0][1][{i}][{j}]");
        }
    }
}

#[test]
fn iota81_needs_no_transpose_for_consecutive_blocks() {
    // The plain reshape already groups 1..9 / 10..18 / ... into
    // the inner matrices -- transpose_axes with identity keeps it.
    let mut env = Environment::new();
    let a = run(
        &mut env,
        "transpose_axes(reshape(range(81), [3, 3, 3, 3]), [0, 1, 2, 3])",
    )
    .unwrap();
    assert_eq!(&a.data()[0..9], &[0., 1., 2., 3., 4., 5., 6., 7., 8.]);
}

#[test]
fn bad_perms_error_loudly() {
    let mut env = Environment::new();
    let e = run(&mut env, "transpose_axes(reshape(range(6), [2, 3]), [0])").unwrap_err();
    assert!(e.contains("transpose_axes"), "length: {e}");
    let e = run(
        &mut env,
        "transpose_axes(reshape(range(6), [2, 3]), [0, 0])",
    )
    .unwrap_err();
    assert!(e.contains("transpose_axes"), "duplicate axis: {e}");
    let e = run(
        &mut env,
        "transpose_axes(reshape(range(6), [2, 3]), [0, 2])",
    )
    .unwrap_err();
    assert!(e.contains("transpose_axes"), "out of range: {e}");
}
