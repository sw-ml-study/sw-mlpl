//! Tape lowering for the non-linear layer lookups: per-row RMS
//! normalization, one-hot-matmul embedding lookup, and the Engram
//! selection-matmul retrieval (saga E2 step 3). All three ride the
//! same trick -- a 0/1 selection matrix matmul'd against the
//! trainable table, whose backward pass IS the scatter-ADD
//! gradient (duplicate addresses accumulate) with unaddressed rows
//! untouched. Bundled here to keep the crate under the 7-module
//! Crate-Module-Count budget.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
use mlpl_engram_core::{HashSpec, head_offset, ngram_hashes};

use crate::error::TapeError;
use crate::linear::linear_tape;

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

/// Named-field inputs for [`engram_tape`]: the five parameter
/// names of a `ModelSpec::Engram` plus its frozen hash spec.
pub struct EngramInputs<'a> {
    pub memory: &'a str,
    pub w_value: &'a str,
    pub b_value: &'a str,
    pub w_gate: &'a str,
    pub b_gate: &'a str,
    pub hash: HashSpec,
    pub head_dim: usize,
    pub hidden: usize,
}

/// `apply_engram(e, h, ids)` on the tape. The addressed rows are
/// gathered by a one-hot selection matmul against the memory table
/// (backward = scatter-ADD into exactly those rows), then reshaped
/// to `[T, retrieved]` and run through the same value/gate linears
/// as the eager forward: `out = h + sigmoid([h|v] @ Wg + bg) * v`.
///
/// Seam decision (saga E5 step 004): the selection matmul IS the
/// device-resident path -- `sel @ memory` gathers and
/// `sel^T @ upstream` is an EXACT scatter-ADD (duplicate addresses
/// accumulate), both lazy on the backend. A "real" device gather
/// was evaluated and rejected: the vendored mlx-rs exposes only
/// scatter-REPLACE (IndexMut), so a `take`-based forward would
/// need a CPU backward and break residency. The dense `[n, rows]`
/// one-hot and its per-step upload are noise at this scale
/// (~0.3% of a 7ms step, 1 of ~350 device interactions); sparse
/// gather/scatter for 100M-row tables is the sparse-rows saga's
/// scope, gated on a true scatter-add kernel.
pub fn engram_tape(
    h: &Tensor,
    ids: &DenseArray,
    inputs: &EngramInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let dims = h.value().shape().dims().to_vec();
    if dims.len() != 2 || dims[1] != inputs.hidden {
        return Err(TapeError::Unsupported(format!(
            "apply_engram (tape): hidden state must be [T, {}], got shape {dims:?}",
            inputs.hidden
        )));
    }
    let id_vals = engram_ids(ids, dims[0])?;
    let hashes =
        ngram_hashes(&id_vals, &inputs.hash).map_err(|e| TapeError::Unsupported(e.to_string()))?;
    let memory_t = params
        .get(inputs.memory)
        .cloned()
        .ok_or_else(|| TapeError::UndefinedVariable(inputs.memory.into()))?;
    let rows = memory_t.value().shape().dims()[0];
    let sel = Tensor::leaf(
        Rc::clone(tape),
        selection_matrix(&hashes, &inputs.hash, rows)?,
        false,
    );
    let width = inputs.hash.ngram_orders.len() * inputs.hash.heads_per_ngram * inputs.head_dim;
    let retrieved = sel
        .matmul(&memory_t)
        .reshape(Shape::new(vec![dims[0], width]));
    let v = linear_tape(&retrieved, inputs.w_value, inputs.b_value, tape, params)?;
    let hv = h.concat(&v, 1);
    let g = linear_tape(&hv, inputs.w_gate, inputs.b_gate, tape, params)?.sigmoid();
    Ok(h.add(&g.mul(&v)))
}

/// Validate `ids` as a rank-1 `[T]` array of non-negative integers.
fn engram_ids(ids: &DenseArray, t_len: usize) -> Result<Vec<u64>, TapeError> {
    let dims = ids.shape().dims();
    if dims.len() != 1 || dims[0] != t_len {
        return Err(TapeError::Unsupported(format!(
            "apply_engram (tape): ids must be rank-1 with the same T as the hidden \
             state ({t_len}), got shape {dims:?}"
        )));
    }
    ids.data()
        .iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 {
                Err(TapeError::Unsupported(format!(
                    "apply_engram (tape): ids must be non-negative integers, got {v}"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect()
}

/// One-hot `[T*orders*heads, rows]` selection matrix: row `r` of
/// the output picks the global memory row addressed by the r-th
/// (t, order, head) triple, in the same layout the eager forward's
/// gather uses.
fn selection_matrix(
    hashes: &[Vec<Vec<u64>>],
    spec: &HashSpec,
    rows: usize,
) -> Result<DenseArray, TapeError> {
    let per_t = spec.ngram_orders.len() * spec.heads_per_ngram;
    let n = hashes.len() * per_t;
    let mut data = vec![0.0_f64; n * rows];
    let mut r = 0;
    for t_hashes in hashes {
        for (oi, order_hashes) in t_hashes.iter().enumerate() {
            for (head, &local) in order_hashes.iter().enumerate() {
                let row = (head_offset(spec, oi, head) + local) as usize;
                data[r * rows + row] = 1.0;
                r += 1;
            }
        }
    }
    DenseArray::new(Shape::new(vec![n, rows]), data).map_err(|e| {
        TapeError::Unsupported(format!(
            "apply_engram (tape): selection matrix construction failed: {e}"
        ))
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
