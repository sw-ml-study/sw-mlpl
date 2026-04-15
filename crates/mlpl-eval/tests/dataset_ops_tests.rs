//! Tests for the dataset-prep builtins: shuffle, batch, batch_mask,
//! split, val_split (Saga 12 step 002).

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

// -- shuffle --

#[test]
fn shuffle_is_a_row_permutation() {
    // [[1,2],[3,4],[5,6],[7,8]] shuffled yields a [4, 2] that
    // contains each original row exactly once.
    let arr = eval("shuffle(reshape(iota(8), [4, 2]), 7)");
    assert_eq!(arr.shape().dims(), &[4, 2]);
    let mut rows: Vec<Vec<f64>> = (0..4)
        .map(|i| arr.data()[i * 2..(i + 1) * 2].to_vec())
        .collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        rows,
        vec![
            vec![0.0, 1.0],
            vec![2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0],
        ]
    );
}

#[test]
fn shuffle_is_deterministic_given_seed() {
    let a = eval("shuffle(iota(10), 42)");
    let b = eval("shuffle(iota(10), 42)");
    assert_eq!(a.data(), b.data());
}

#[test]
fn shuffle_different_seeds_differ() {
    // Not strictly guaranteed for every pair, but for these small
    // shuffles the xorshift output differs across seeds.
    let a = eval("shuffle(iota(10), 1)");
    let b = eval("shuffle(iota(10), 999)");
    assert_ne!(a.data(), b.data());
}

#[test]
fn shuffle_preserves_non_axis0_labels() {
    // label axis 0 and axis 1; shuffle should drop axis-0 label
    // (rows are permuted, so the original axis-0 name no longer
    // makes sense) and keep axis-1's.
    let arr = eval(
        "x : [batch, feat] = reshape(iota(6), [3, 2])\n\
         shuffle(x, 7)",
    );
    assert_eq!(arr.labels(), Some(&[None, Some("feat".into())][..]));
}

// -- batch --

#[test]
fn batch_reshapes_with_zero_padding() {
    // N=5, size=2 -> [3, 2] batches. Last batch padded with 0.
    let arr = eval("batch(iota(5), 2)");
    assert_eq!(arr.shape().dims(), &[3, 2]);
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
}

#[test]
fn batch_exact_division_no_padding() {
    let arr = eval("batch(iota(6), 3)");
    assert_eq!(arr.shape().dims(), &[2, 3]);
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn batch_rank2_makes_rank3() {
    // [4, 2] batched with size 2 -> [2, 2, 2]
    let arr = eval("batch(reshape(iota(8), [4, 2]), 2)");
    assert_eq!(arr.shape().dims(), &[2, 2, 2]);
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

// -- batch_mask --

#[test]
fn batch_mask_flags_valid_rows() {
    // N=5, size=2 -> [3, 2] mask: [[1, 1], [1, 1], [1, 0]]
    let arr = eval("batch_mask(iota(5), 2)");
    assert_eq!(arr.shape().dims(), &[3, 2]);
    assert_eq!(arr.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn batch_mask_exact_division_all_ones() {
    let arr = eval("batch_mask(iota(6), 3)");
    assert_eq!(
        arr,
        DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6]).unwrap()
    );
}

// -- split / val_split --

#[test]
fn split_and_val_split_are_disjoint_and_cover() {
    // 10 rows, frac 0.8 -> train has 8, val has 2. Together they
    // contain every original row exactly once.
    let train = eval("split(iota(10), 0.8, 7)");
    let val = eval("val_split(iota(10), 0.8, 7)");
    assert_eq!(train.shape().dims(), &[8]);
    assert_eq!(val.shape().dims(), &[2]);
    let mut all: Vec<f64> = train.data().to_vec();
    all.extend_from_slice(val.data());
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(all, (0..10).map(|i| i as f64).collect::<Vec<_>>());
}

#[test]
fn split_deterministic_across_runs() {
    let a = eval("split(iota(20), 0.7, 42)");
    let b = eval("split(iota(20), 0.7, 42)");
    assert_eq!(a.data(), b.data());
}

#[test]
fn split_preserves_non_axis0_labels() {
    let train = eval(
        "x : [batch, feat] = reshape(iota(12), [6, 2])\n\
         split(x, 0.5, 3)",
    );
    assert_eq!(train.shape().dims(), &[3, 2]);
    assert_eq!(train.labels(), Some(&[None, Some("feat".into())][..]));
}

#[test]
fn split_train_frac_zero_errors() {
    let src = "split(iota(10), 0.0, 7)";
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let r = eval_program(&stmts, &mut env);
    assert!(r.is_err(), "train_frac=0 should error");
}

#[test]
fn split_train_frac_one_errors() {
    let src = "split(iota(10), 1.0, 7)";
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let r = eval_program(&stmts, &mut env);
    assert!(r.is_err(), "train_frac=1 should error");
}
