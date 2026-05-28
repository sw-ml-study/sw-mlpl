//! Per-variant clone helpers extracted from `clone.rs`. Each
//! takes the variant fields, mints fresh layer ids, and copies
//! params + device tags.

use mlpl_env_traits::{HasModelIds, HasParams, HasTensorDevices, HasVars};
use mlpl_eval_core::model::ModelSpec;

use crate::error::MutateError;

pub(crate) fn clone_chain<E>(children: &[ModelSpec], env: &mut E) -> Result<ModelSpec, MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let mut out = Vec::with_capacity(children.len());
    for child in children {
        out.push(crate::clone::clone_spec(child, env)?);
    }
    Ok(ModelSpec::Chain(out))
}

pub(crate) fn clone_linear<E>(env: &mut E, w: &str, b: &str) -> Result<ModelSpec, MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let id = env.alloc_model_id();
    let new_w = format!("__linear_W_{id}");
    let new_b = format!("__linear_b_{id}");
    copy_param(env, w, &new_w)?;
    copy_param(env, b, &new_b)?;
    Ok(ModelSpec::Linear { w: new_w, b: new_b })
}

pub(crate) fn clone_embedding<E>(
    env: &mut E,
    table: &str,
    vocab: usize,
    d_model: usize,
) -> Result<ModelSpec, MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let id = env.alloc_model_id();
    let new_table = format!("__embed_E_{id}");
    copy_param(env, table, &new_table)?;
    Ok(ModelSpec::Embedding {
        table: new_table,
        vocab,
        d_model,
    })
}

pub(crate) fn copy_param<E>(env: &mut E, old: &str, new: &str) -> Result<(), MutateError>
where
    E: HasVars + HasParams + HasTensorDevices,
{
    let value = env
        .get(old)
        .cloned()
        .ok_or_else(|| MutateError::UndefinedVariable(old.into()))?;
    let device = env.tensor_device(old).to_string();
    env.set_param(new.to_string(), value);
    env.set_tensor_device(new.to_string(), device);
    Ok(())
}
