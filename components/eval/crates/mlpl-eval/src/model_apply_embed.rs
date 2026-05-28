//! Saga 33 step 004: embedding-lookup input prep extracted from
//! `model_dispatch.rs`. `tokens_to_onehot` converts a 1-D integer
//! token-id array into a `[N, vocab]` one-hot matrix so the
//! embedding lookup can lower to a plain matmul -- giving us
//! autograd through the embedding table for free.

use mlpl_array::{DenseArray, Shape};

use mlpl_eval_types::EvalError;

/// Convert an integer-valued token id array (1-D `[N]`) into a
/// `[N, vocab]` one-hot matrix. Token ids must be non-negative
/// integers in `[0, vocab)`.
pub(crate) fn tokens_to_onehot(tokens: &DenseArray, vocab: usize) -> Result<DenseArray, EvalError> {
    let dims = tokens.shape().dims();
    if dims.len() != 1 {
        let msg = format!("embed: tokens must be a 1-D [N] array, got shape {dims:?}");
        return Err(EvalError::Unsupported(msg));
    }
    let n = dims[0];
    let mut data = vec![0.0_f64; n * vocab];
    for (row, &id_f) in tokens.data().iter().enumerate() {
        let id = validate_token_id(row, id_f, vocab)?;
        data[row * vocab + id] = 1.0;
    }
    DenseArray::new(Shape::new(vec![n, vocab]), data)
        .map_err(|e| EvalError::Unsupported(format!("embed: one-hot construction failed: {e}")))
}

fn validate_token_id(row: usize, id_f: f64, vocab: usize) -> Result<usize, EvalError> {
    if !id_f.is_finite() || id_f < 0.0 || id_f.fract() != 0.0 {
        let msg = format!("embed: token at position {row} = {id_f} is not a non-negative integer");
        return Err(EvalError::Unsupported(msg));
    }
    let id = id_f as usize;
    if id >= vocab {
        let msg = format!("embed: token at position {row} = {id} out of vocab range [0, {vocab})");
        return Err(EvalError::Unsupported(msg));
    }
    Ok(id)
}
