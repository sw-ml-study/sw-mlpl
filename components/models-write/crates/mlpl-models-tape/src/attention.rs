//! Scaled-dot-product attention dispatcher. Handles the rank
//! dispatch (rank-2 `[T, d]` vs rank-3 `[B, T, d]`) and the
//! per-head context computation delegates to
//! `attention_heads`.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::Shape;
use mlpl_autograd::{Tape, Tensor};

use crate::attention_heads::{ProjectedQKV, multi_head_context, single_head_context};
use crate::error::TapeError;

/// Bundle of attention parameter names + model dimension so the
/// tape helper stays inside the 7-arg clippy budget.
pub struct AttentionInputs<'a> {
    pub wq: &'a str,
    pub wk: &'a str,
    pub wv: &'a str,
    pub wo: &'a str,
    pub d_model: usize,
    pub heads: usize,
    pub causal: bool,
}

pub fn attention_tape(
    x: &Tensor,
    inputs: &AttentionInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let dims = x.value().shape().dims().to_vec();
    if dims.len() == 3 && dims[2] == inputs.d_model {
        return attention_tape_rank3(x, inputs, tape, params, dims);
    }
    if dims.len() != 2 || dims[1] != inputs.d_model {
        return Err(TapeError::Unsupported(format!(
            "attention: input must be [seq, {}] or [B, T, {}], got {dims:?}",
            inputs.d_model, inputs.d_model
        )));
    }
    attention_tape_rank2(x, inputs, tape, params)
}

fn attention_tape_rank2(
    x: &Tensor,
    inputs: &AttentionInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let qkv = ProjectedQKV::project(x, inputs, params)?;
    let wo_t = fetch(params, inputs.wo)?;
    let context = if inputs.heads == 1 {
        single_head_context(&qkv, inputs.d_model, inputs.causal, tape)?
    } else {
        multi_head_context(&qkv, inputs, tape)?
    };
    Ok(context.matmul(&wo_t))
}

fn attention_tape_rank3(
    x: &Tensor,
    inputs: &AttentionInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
    dims: Vec<usize>,
) -> Result<Tensor, TapeError> {
    let (batch, t, d) = (dims[0], dims[1], dims[2]);
    if batch == 0 {
        return Err(TapeError::Unsupported(
            "attention: rank-3 input had zero batch entries".into(),
        ));
    }
    let mut per_batch: Vec<Tensor> = Vec::with_capacity(batch);
    for b in 0..batch {
        let x_b = x.take(0, b);
        let out_b = attention_tape_rank2(&x_b, inputs, tape, params)?;
        per_batch.push(out_b.reshape(Shape::new(vec![1, t, d])));
    }
    Ok(Tensor::stack(&per_batch, 0))
}

pub(crate) fn fetch(params: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, TapeError> {
    params
        .get(name)
        .cloned()
        .ok_or_else(|| TapeError::UndefinedVariable(name.into()))
}
