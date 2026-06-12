//! Per-step plumbing for the CUDA LoRA path (see `grad_optim_cuda`):
//! one-hot inputs, the frozen-weight bundle, and the on-device adam
//! update with moments persisted in `env.optim_state`. The CUDA analog
//! of `grad_optim_mlx_step`.

use crate::gpu_step::GpuEnv;
use crate::grad_optim_mlx_demo::DemoLayout;
use crate::model_apply_embed::tokens_to_onehot;
use candle_core::Tensor;
use mlpl_array::DenseArray;
use mlpl_cuda_forward::causal_mask;
use mlpl_cuda_model::DemoWeights;
use mlpl_cuda_rt::{cuda_device, cuda_to_dense_data, dense_to_cuda};
use mlpl_cuda_train::{AdamHp, adam_update};
use mlpl_eval_types::EvalError;

/// Lower integer token ids `[N]` to a one-hot `[N, vocab]` candle tensor.
pub(crate) fn tokens_cuda(tokens: &DenseArray, vocab: usize) -> Result<Tensor, EvalError> {
    let oh = tokens_to_onehot(tokens, vocab)?;
    Ok(dense_to_cuda(oh.data(), oh.shape().dims()))
}

/// Frozen base weights as candle tensors; gamma is all-ones (RMSNorm is
/// gamma-free) and the mask is sized to the token count `seq`.
pub(crate) fn build_weights(
    layout: &DemoLayout,
    env: &dyn GpuEnv,
    seq: usize,
) -> Result<DemoWeights, EvalError> {
    let get = |n: &str| -> Result<Tensor, EvalError> {
        let d = env
            .binding(n)
            .ok_or_else(|| EvalError::UndefinedVariable(n.into()))?;
        Ok(dense_to_cuda(d.data(), d.shape().dims()))
    };
    let d_model = env.binding(&layout.wq).map_or(0, |a| a.shape().dims()[0]);
    let mask = causal_mask(seq, cuda_device())
        .map_err(|e| EvalError::Unsupported(format!("cuda lora mask: {e}")))?;
    Ok(DemoWeights {
        embed: get(&layout.embed_table)?,
        wq: get(&layout.wq)?,
        wk: get(&layout.wk)?,
        wv: get(&layout.wv)?,
        wo: get(&layout.wo)?,
        head_w: get(&layout.head_w)?,
        head_b: get(&layout.head_b)?,
        gamma: dense_to_cuda(&vec![1.0; d_model], &[d_model]),
        mask,
        scale: layout.alpha / layout.rank as f64,
        eps: 1e-8,
    })
}

/// One on-device adam step for `param`: read its moments from
/// `env.optim_state` (keyed like the CPU path), update, and write the
/// new weight + moments back into the Environment.
pub(crate) fn step_adapter(
    env: &mut dyn GpuEnv,
    param: &str,
    grad: &Tensor,
    hp: &AdamHp,
) -> Result<(), EvalError> {
    let dims = env.binding(param).expect("adapter present").shape().clone();
    let cur = dense_to_cuda(env.binding(param).unwrap().data(), dims.dims());
    let load = |suf: &str, e: &dyn GpuEnv| {
        e.optim_buffer("adam", param, suf).map_or_else(
            || dense_to_cuda(&vec![0.0; dims.elem_count()], dims.dims()),
            |d| dense_to_cuda(d.data(), d.shape().dims()),
        )
    };
    let (m, v) = (load("m", env), load("v", env));
    let (w_new, m_new, v_new) = adam_update(&cur, grad, &m, &v, hp)
        .map_err(|e| EvalError::Unsupported(format!("cuda lora: {e}")))?;
    let flat = |a: &Tensor| DenseArray::new(dims.clone(), cuda_to_dense_data(a));
    env.set_binding(param.to_string(), flat(&w_new)?);
    env.set_optim_buffer("adam", param, "m", flat(&m_new)?);
    env.set_optim_buffer("adam", param, "v", flat(&v_new)?);
    Ok(())
}
