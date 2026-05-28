//! Per-head attention context computation. Single-head is a
//! straight `softmax(scale * Q K^T + mask) V`; multi-head splits
//! Q/K/V into per-head slabs via `reshape + take` and stacks the
//! per-head outputs.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

use crate::attention::{AttentionInputs, fetch};
use crate::error::TapeError;

/// Pre-projected Q / K / V tensors for one attention call.
pub(crate) struct ProjectedQKV {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

impl ProjectedQKV {
    pub(crate) fn project(
        x: &Tensor,
        inputs: &AttentionInputs<'_>,
        params: &HashMap<String, Tensor>,
    ) -> Result<Self, TapeError> {
        Ok(Self {
            q: x.matmul(&fetch(params, inputs.wq)?),
            k: x.matmul(&fetch(params, inputs.wk)?),
            v: x.matmul(&fetch(params, inputs.wv)?),
        })
    }
}

/// Single-head context: `softmax(scale * Q @ K^T + mask) @ V`.
pub(crate) fn single_head_context(
    qkv: &ProjectedQKV,
    d_model: usize,
    causal: bool,
    tape: &Rc<Tape>,
) -> Result<Tensor, TapeError> {
    let scale = Tensor::leaf(
        Rc::clone(tape),
        DenseArray::from_scalar(1.0 / (d_model as f64).sqrt()),
        false,
    );
    let scores = qkv.q.matmul(&qkv.k.transpose()).mul(&scale);
    let masked = if causal {
        let seq = scores.value().shape().dims()[0];
        scores.add(&Tensor::leaf(
            Rc::clone(tape),
            causal_mask_array(seq)?,
            false,
        ))
    } else {
        scores
    };
    Ok(masked.softmax().matmul(&qkv.v))
}

/// Multi-head context: split Q/K/V into `heads` slabs of width
/// `d_k = d_model / heads`, run per-head scaled-dot-product on
/// each, stack outputs over the head axis.
pub(crate) fn multi_head_context(
    qkv: &ProjectedQKV,
    inputs: &AttentionInputs<'_>,
    tape: &Rc<Tape>,
) -> Result<Tensor, TapeError> {
    if !inputs.d_model.is_multiple_of(inputs.heads) {
        return Err(TapeError::Unsupported(format!(
            "attention: d_model ({}) must be divisible by heads ({})",
            inputs.d_model, inputs.heads
        )));
    }
    let d_k = inputs.d_model / inputs.heads;
    let t = qkv.q.value().shape().dims()[0];
    let head_shape = Shape::new(vec![t, inputs.heads, d_k]);
    let q_split = qkv.q.reshape(head_shape.clone());
    let k_split = qkv.k.reshape(head_shape.clone());
    let v_split = qkv.v.reshape(head_shape);
    let scale = Tensor::leaf(
        Rc::clone(tape),
        DenseArray::from_scalar(1.0 / (d_k as f64).sqrt()),
        false,
    );
    let mask = if inputs.causal {
        Some(Tensor::leaf(Rc::clone(tape), causal_mask_array(t)?, false))
    } else {
        None
    };
    let head_outs: Vec<Tensor> = (0..inputs.heads)
        .map(|h| {
            let q_h = q_split.take(1, h);
            let k_h = k_split.take(1, h);
            let v_h = v_split.take(1, h);
            let scores = q_h.matmul(&k_h.transpose()).mul(&scale);
            let masked = match &mask {
                Some(m) => scores.add(m),
                None => scores,
            };
            masked.softmax().matmul(&v_h)
        })
        .collect();
    Ok(Tensor::stack(&head_outs, 1))
}

fn causal_mask_array(seq: usize) -> Result<DenseArray, TapeError> {
    let neg = -1.0e9_f64;
    let mut data = vec![0.0_f64; seq * seq];
    for r in 0..seq {
        for c in (r + 1)..seq {
            data[r * seq + c] = neg;
        }
    }
    Ok(DenseArray::new(Shape::new(vec![seq, seq]), data)?)
}
