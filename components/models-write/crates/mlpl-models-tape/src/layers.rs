//! Tape lowering for the parameter-light layer ops: per-row RMS
//! normalization and one-hot-matmul embedding lookup. Both are
//! small enough that bundling them keeps the crate under the
//! 7-module Crate-Module-Count budget.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

use crate::error::TapeError;

/// Per-row RMS normalization on the tape:
/// `y[i, :] = x[i, :] / sqrt(mean(x[i, :]^2) + eps)`.
///
/// `sqrt(v)` is encoded as `exp(-0.5 * log(v))` because the tape
/// doesn't expose a direct sqrt op. Per-row mean is encoded as
/// `matmul(x, ones([cols, 1])) / cols`, and the broadcast back
/// to `[rows, cols]` as `rsqrt @ ones([1, cols])`.
pub fn rms_norm_tape(x: &Tensor, tape: &Rc<Tape>) -> Result<Tensor, TapeError> {
    let dims = x.value().shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(TapeError::Unsupported(
            "rms_norm: input must be a rank-2 [rows, cols] matrix".into(),
        ));
    }
    let cols = dims[1];
    let eps = 1e-8_f64;
    let leaf = |v: DenseArray| Tensor::leaf(Rc::clone(tape), v, false);
    let ones_col = leaf(DenseArray::new(Shape::new(vec![cols, 1]), vec![1.0; cols])?);
    let ones_row = leaf(DenseArray::new(Shape::new(vec![1, cols]), vec![1.0; cols])?);
    let inv_cols = leaf(DenseArray::from_scalar(1.0 / cols as f64));
    let eps_t = leaf(DenseArray::from_scalar(eps));
    let half_neg = leaf(DenseArray::from_scalar(-0.5));
    let row_mean_eps = x.mul(x).matmul(&ones_col).mul(&inv_cols).add(&eps_t);
    let rsqrt = row_mean_eps.log().mul(&half_neg).exp();
    Ok(x.mul(&rsqrt.matmul(&ones_row)))
}

/// Embedding lookup on the tape. The token id array enters as a
/// non-trainable input leaf; its eager value builds a one-hot
/// matrix which then matmuls against the trainable table tensor,
/// routing backprop straight into the table's gradient buffer.
pub fn embedding_tape(
    x: &Tensor,
    table: &str,
    vocab: usize,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let table_t = params
        .get(table)
        .cloned()
        .ok_or_else(|| TapeError::UndefinedVariable(table.into()))?;
    let tokens_arr = x.value().clone();
    let onehot_arr = onehot_from_tokens(&tokens_arr, vocab)?;
    let onehot_t = Tensor::leaf(Rc::clone(tape), onehot_arr, false);
    Ok(onehot_t.matmul(&table_t))
}

/// One-hot encode a 1-D token id array `[N]` into `[N, vocab]`.
fn onehot_from_tokens(tokens: &DenseArray, vocab: usize) -> Result<DenseArray, TapeError> {
    let dims = tokens.shape().dims();
    if dims.len() != 1 {
        let msg = format!("embed (tape): tokens must be a 1-D [N] array, got shape {dims:?}");
        return Err(TapeError::Unsupported(msg));
    }
    let n = dims[0];
    let mut data = vec![0.0_f64; n * vocab];
    for (row, &id_f) in tokens.data().iter().enumerate() {
        let id = validate_token_id(row, id_f, vocab)?;
        data[row * vocab + id] = 1.0;
    }
    DenseArray::new(Shape::new(vec![n, vocab]), data).map_err(|e| {
        TapeError::Unsupported(format!("embed (tape): one-hot construction failed: {e}"))
    })
}

fn validate_token_id(row: usize, id_f: f64, vocab: usize) -> Result<usize, TapeError> {
    if !id_f.is_finite() || id_f < 0.0 || id_f.fract() != 0.0 {
        let msg =
            format!("embed (tape): token at position {row} = {id_f} is not a non-negative integer");
        return Err(TapeError::Unsupported(msg));
    }
    let id = id_f as usize;
    if id >= vocab {
        let msg =
            format!("embed (tape): token at position {row} = {id} out of vocab range [0, {vocab})");
        return Err(TapeError::Unsupported(msg));
    }
    Ok(id)
}
