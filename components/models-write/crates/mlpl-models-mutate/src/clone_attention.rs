//! Attention + LinearLora clone helpers extracted from
//! `clone.rs`. These two variants have larger field counts so
//! their helpers live in this dedicated file to keep both
//! `clone.rs` and `clone_variants.rs` below the 4-fn warn
//! line.

use mlpl_env_traits::{HasModelIds, HasParams, HasTensorDevices, HasVars};
use mlpl_eval_core::model::ModelSpec;

use crate::clone_variants::copy_param;
use crate::error::MutateError;

pub(crate) fn clone_attention<E>(spec: &ModelSpec, env: &mut E) -> Result<ModelSpec, MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let ModelSpec::Attention {
        wq,
        wk,
        wv,
        wo,
        d_model,
        heads,
        causal,
    } = spec
    else {
        unreachable!("clone_attention on non-Attention");
    };
    let id = env.alloc_model_id();
    let names = ["Wq", "Wk", "Wv", "Wo"].map(|tag| format!("__attn_{tag}_{id}"));
    for (old, new) in [wq, wk, wv, wo].iter().zip(names.iter()) {
        copy_param(env, old, new)?;
    }
    let [new_wq, new_wk, new_wv, new_wo] = names;
    Ok(ModelSpec::Attention {
        wq: new_wq,
        wk: new_wk,
        wv: new_wv,
        wo: new_wo,
        d_model: *d_model,
        heads: *heads,
        causal: *causal,
    })
}

pub(crate) fn clone_linear_lora<E>(spec: &ModelSpec, env: &mut E) -> Result<ModelSpec, MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let ModelSpec::LinearLora {
        w,
        b,
        a,
        b_adapter,
        in_dim,
        out_dim,
        rank,
        alpha,
    } = spec
    else {
        unreachable!("clone_linear_lora on non-LinearLora");
    };
    let (new_w, new_b) = mint_lora_base(env, w, b)?;
    let (new_a, new_b_adapter) = mint_lora_adapter(env, a, b_adapter)?;
    Ok(ModelSpec::LinearLora {
        w: new_w,
        b: new_b,
        a: new_a,
        b_adapter: new_b_adapter,
        in_dim: *in_dim,
        out_dim: *out_dim,
        rank: *rank,
        alpha: *alpha,
    })
}

fn mint_lora_base<E>(env: &mut E, w: &str, b: &str) -> Result<(String, String), MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let id = env.alloc_model_id();
    let new_w = format!("__linear_W_{id}");
    let new_b = format!("__linear_b_{id}");
    copy_param(env, w, &new_w)?;
    copy_param(env, b, &new_b)?;
    Ok((new_w, new_b))
}

fn mint_lora_adapter<E>(
    env: &mut E,
    a: &str,
    b_adapter: &str,
) -> Result<(String, String), MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let id = env.alloc_model_id();
    let new_a = format!("__lora_A_{id}");
    let new_b_adapter = format!("__lora_B_{id}");
    copy_param(env, a, &new_a)?;
    copy_param(env, b_adapter, &new_b_adapter)?;
    Ok((new_a, new_b_adapter))
}
