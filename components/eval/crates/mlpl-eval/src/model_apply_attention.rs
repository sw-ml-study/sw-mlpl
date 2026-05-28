//! Saga 33 step 004: multi-head attention forward pass extracted
//! from `model_dispatch.rs`. `apply_attention` is the top-level
//! entry that handles rank-2 [seq, d_model] directly and rank-3
//! [B, T, d_model] by looping over the batch axis. Per-head work
//! lives in `apply_attn_head`; column slicing in `slice_cols`.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::prelude::*;

use crate::env::Environment;
use crate::error::EvalError;

/// Bundle of `Attention` layer parameters threaded through
/// `apply_attention` and its forward-pass helpers. Replaces a
/// 9-argument signature that previously needed
/// `#[allow(clippy::too_many_arguments)]`.
pub(crate) struct AttentionArgs<'a> {
    pub wq: &'a str,
    pub wk: &'a str,
    pub wv: &'a str,
    pub wo: &'a str,
    pub d_model: usize,
    pub heads: usize,
    pub causal: bool,
}

/// Pre-computed per-attention-call context shared across every
/// head: the three projected matrices Q / K / V, the geometry
/// (`d_k`, `d_model`, `seq`), the score scale factor, and the
/// causal-mask flag.
struct AttnHeadCtx<'a> {
    q: &'a DenseArray,
    k: &'a DenseArray,
    v: &'a DenseArray,
    d_k: usize,
    d_model: usize,
    seq: usize,
    scale: f64,
    causal: bool,
}

pub(crate) fn apply_attention(
    x: &DenseArray,
    args: &AttentionArgs<'_>,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    let d_model = args.d_model;
    if dims.len() == 3 && dims[2] == d_model {
        let (batch, t) = (dims[0], dims[1]);
        let mut data = Vec::with_capacity(batch * t * d_model);
        for b in 0..batch {
            let x_b = x.take(0, b)?;
            data.extend_from_slice(apply_attention_rank2(&x_b, args, env)?.data());
        }
        return Ok(DenseArray::new(Shape::new(vec![batch, t, d_model]), data)?);
    }
    if dims.len() != 2 || dims[1] != d_model {
        return Err(EvalError::Unsupported(format!(
            "attention: input must be [seq, {d_model}] or [B, T, {d_model}], got {:?}",
            dims
        )));
    }
    apply_attention_rank2(x, args, env)
}

fn apply_attention_rank2(
    x: &DenseArray,
    args: &AttentionArgs<'_>,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    let d_model = args.d_model;
    let seq = dims[0];
    let d_k = d_model / args.heads;
    let lookup = |n: &str| -> Result<DenseArray, EvalError> {
        env.get(n)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(n.into()))
    };
    let project = |w: DenseArray| crate::device::dispatched_call(env, "matmul", vec![x.clone(), w]);
    let q = project(lookup(args.wq)?)?;
    let k = project(lookup(args.wk)?)?;
    let v = project(lookup(args.wv)?)?;
    let wo_a = lookup(args.wo)?;
    let scale = 1.0 / (d_k as f64).sqrt();
    let ctx = AttnHeadCtx {
        q: &q,
        k: &k,
        v: &v,
        d_k,
        d_model,
        seq,
        scale,
        causal: args.causal,
    };
    let mut concat = vec![0.0_f64; seq * d_model];
    for h in 0..args.heads {
        apply_attn_head(&ctx, h, env, &mut concat)?;
    }
    let concat = DenseArray::new(Shape::new(vec![seq, d_model]), concat)?;
    crate::device::dispatched_call(env, "matmul", vec![concat, wo_a])
}

fn apply_attn_head(
    ctx: &AttnHeadCtx<'_>,
    h: usize,
    env: &Environment,
    concat: &mut [f64],
) -> Result<(), EvalError> {
    let q_h = slice_cols(ctx.q, h * ctx.d_k, ctx.d_k)?;
    let k_h = slice_cols(ctx.k, h * ctx.d_k, ctx.d_k)?;
    let v_h = slice_cols(ctx.v, h * ctx.d_k, ctx.d_k)?;
    let kt = crate::device::dispatched_call(env, "transpose", vec![k_h])?;
    let scores = crate::device::dispatched_call(env, "matmul", vec![q_h, kt])?;
    let seq = ctx.seq;
    let causal = ctx.causal;
    let scale = ctx.scale;
    let scaled: Vec<f64> = scores
        .data()
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if causal && i % seq > i / seq {
                -1.0e9
            } else {
                s * scale
            }
        })
        .collect();
    let scores_scaled = DenseArray::new(Shape::new(vec![seq, seq]), scaled)?;
    let attn = crate::device::dispatched_call(
        env,
        "softmax",
        vec![scores_scaled, DenseArray::from_scalar(1.0)],
    )?;
    let head_out = crate::device::dispatched_call(env, "matmul", vec![attn, v_h])?;
    for r in 0..seq {
        for c in 0..ctx.d_k {
            concat[r * ctx.d_model + h * ctx.d_k + c] = head_out.data()[r * ctx.d_k + c];
        }
    }
    Ok(())
}

/// Extract `width` consecutive columns starting at `start` from a
/// rank-2 matrix.
pub(crate) fn slice_cols(
    x: &DenseArray,
    start: usize,
    width: usize,
) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    let rows = dims[0];
    let cols = dims[1];
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        for c in 0..width {
            out.push(x.data()[r * cols + start + c]);
        }
    }
    Ok(DenseArray::new(Shape::new(vec![rows, width]), out)?)
}
