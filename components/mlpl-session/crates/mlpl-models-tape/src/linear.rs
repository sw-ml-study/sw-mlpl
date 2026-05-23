//! Linear + LinearLora tape lowering. `linear_tape` emits
//! `x @ W + bias_broadcast(b)`; `linear_lora_tape` emits the
//! same base plus `(alpha/rank) * (x @ A) @ B`.
//!
//! `linear_lora_tape` is composed from small named helpers
//! (`fetch_lora_params`, `lora_adapter`, `bias_broadcast`) so
//! the top-level function stays under the 25-LOC budget.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

use crate::error::TapeError;

/// `linear(W, b)` tape lowering. `b` is `[1, out]`; lifted to
/// `[n, out]` via `ones([n, 1]) @ b` so the tape graph matches
/// the hand-rolled broadcast form exactly.
pub fn linear_tape(
    x: &Tensor,
    w: &str,
    b: &str,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let fetch = |name: &str| {
        params
            .get(name)
            .cloned()
            .ok_or_else(|| TapeError::UndefinedVariable(name.into()))
    };
    let w_t = fetch(w)?;
    let b_t = fetch(b)?;
    let xw = x.matmul(&w_t);
    let bias = bias_broadcast(&xw, &b_t, tape)?;
    Ok(xw.add(&bias))
}

/// Named-field inputs for `linear_lora_tape` so the helper stays
/// at 4 args and dodges `clippy::too_many_arguments`.
pub struct LinearLoraInputs<'a> {
    pub w: &'a str,
    pub b: &'a str,
    pub a: &'a str,
    pub b_adapter: &'a str,
    pub rank: usize,
    pub alpha: f64,
}

/// LoRA tape lowering: `xw + (alpha/rank) * (x @ A @ B) + bias`.
/// Decomposed into `lora_adapter` for the adapter branch +
/// `bias_broadcast` for the bias term so this fn stays under
/// 25 LOC.
pub fn linear_lora_tape(
    x: &Tensor,
    inputs: &LinearLoraInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let w_t = params
        .get(inputs.w)
        .cloned()
        .ok_or_else(|| TapeError::UndefinedVariable(inputs.w.into()))?;
    let b_t = params
        .get(inputs.b)
        .cloned()
        .ok_or_else(|| TapeError::UndefinedVariable(inputs.b.into()))?;
    let xw = x.matmul(&w_t);
    let adapter = lora_adapter(x, inputs, tape, params)?;
    let bias = bias_broadcast(&xw, &b_t, tape)?;
    Ok(xw.add(&adapter).add(&bias))
}

/// Adapter branch: `(alpha/rank) * (x @ A) @ B`.
fn lora_adapter(
    x: &Tensor,
    inputs: &LinearLoraInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let a_t = params
        .get(inputs.a)
        .cloned()
        .ok_or_else(|| TapeError::UndefinedVariable(inputs.a.into()))?;
    let b_adapt_t = params
        .get(inputs.b_adapter)
        .cloned()
        .ok_or_else(|| TapeError::UndefinedVariable(inputs.b_adapter.into()))?;
    let xa = x.matmul(&a_t);
    let xab = xa.matmul(&b_adapt_t);
    let scale = inputs.alpha / inputs.rank as f64;
    let scale_t = Tensor::leaf(Rc::clone(tape), DenseArray::from_scalar(scale), false);
    Ok(xab.mul(&scale_t))
}

/// Broadcast a `[1, out]` bias to `[n, out]` via `ones([n,1]) @ b`,
/// so the tape graph mirrors the eager forward pass exactly.
fn bias_broadcast(xw: &Tensor, b_t: &Tensor, tape: &Rc<Tape>) -> Result<Tensor, TapeError> {
    let n = xw.value().shape().dims()[0];
    let ones_arr = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
    let ones_t = Tensor::leaf(Rc::clone(tape), ones_arr, false);
    Ok(ones_t.matmul(b_t))
}
