//! Per-Linear adapter allocation + the recursive tree
//! rewriter that turns each Linear into a LinearLora. Extracted
//! from `lora.rs` so each file stays under the 4-fn warn line.

use mlpl_array::{DenseArray, Shape};
use mlpl_env_traits::{HasModelIds, HasParams, HasTensorDevices, HasVars};
use mlpl_eval_core::model::ModelSpec;

use crate::error::TuneError;

/// Shared rewrite context threaded through the rewriter.
pub(crate) struct LoraCtx {
    pub rank: usize,
    pub alpha: f64,
    pub seed: f64,
    pub counter: usize,
}

/// Walk the cloned tree and wrap each Linear in a LinearLora.
/// Composite nodes (Chain, Residual) recurse; parameter-bearing
/// non-Linear nodes (Embedding, Attention) and parameter-free
/// nodes (Activation, RmsNorm) are passed through unchanged.
pub(crate) fn rewrite_linears_to_lora<E>(
    spec: ModelSpec,
    env: &mut E,
    ctx: &mut LoraCtx,
) -> Result<ModelSpec, TuneError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    match spec {
        ModelSpec::Linear { w, b } => wrap_linear(env, w, b, ctx),
        ModelSpec::Chain(children) => {
            let out: Vec<_> = children
                .into_iter()
                .map(|c| rewrite_linears_to_lora(c, env, ctx))
                .collect::<Result<_, _>>()?;
            Ok(ModelSpec::Chain(out))
        }
        ModelSpec::Residual(inner) => {
            let wrapped = rewrite_linears_to_lora(*inner, env, ctx)?;
            Ok(ModelSpec::Residual(Box::new(wrapped)))
        }
        ModelSpec::Activation(_)
        | ModelSpec::RmsNorm { .. }
        | ModelSpec::Embedding { .. }
        | ModelSpec::Attention { .. } => Ok(spec),
        ModelSpec::LinearLora { .. } => Err(TuneError::UnexpectedLoraInTree),
    }
}

fn wrap_linear<E>(
    env: &mut E,
    w: String,
    b: String,
    ctx: &mut LoraCtx,
) -> Result<ModelSpec, TuneError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    let w_arr = env
        .get(&w)
        .ok_or_else(|| TuneError::UndefinedVariable(w.clone()))?;
    let dims = w_arr.shape().dims();
    if dims.len() != 2 {
        return Err(TuneError::NonRank2Linear {
            name: w.clone(),
            rank: dims.len(),
        });
    }
    let (in_dim, out_dim) = (dims[0], dims[1]);
    let device = env.tensor_device(&w).to_string();
    let a_seed = ctx.seed + ctx.counter as f64;
    ctx.counter += 1;
    let (a_name, b_adapter_name) =
        allocate_adapters(env, in_dim, out_dim, ctx.rank, &device, a_seed)?;
    Ok(ModelSpec::LinearLora {
        w,
        b,
        a: a_name,
        b_adapter: b_adapter_name,
        in_dim,
        out_dim,
        rank: ctx.rank,
        alpha: ctx.alpha,
    })
}

fn allocate_adapters<E>(
    env: &mut E,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    device: &str,
    a_seed: f64,
) -> Result<(String, String), TuneError>
where
    E: HasParams + HasTensorDevices + HasModelIds,
{
    if rank > in_dim.min(out_dim) {
        return Err(TuneError::RankTooLarge {
            rank,
            in_dim,
            out_dim,
        });
    }
    let id = env.alloc_model_id();
    let a_name = format!("__lora_A_{id}");
    let b_name = format!("__lora_B_{id}");
    let a_arr = make_a_adapter(in_dim, rank, a_seed)?;
    env.set_param(a_name.clone(), a_arr);
    env.set_tensor_device(a_name.clone(), device.into());
    let b_arr = DenseArray::zeros(Shape::new(vec![rank, out_dim]));
    env.set_param(b_name.clone(), b_arr);
    env.set_tensor_device(b_name.clone(), device.into());
    Ok((a_name, b_name))
}

fn make_a_adapter(in_dim: usize, rank: usize, a_seed: f64) -> Result<DenseArray, TuneError> {
    let shape_arr = DenseArray::new(Shape::new(vec![2]), vec![in_dim as f64, rank as f64])?;
    let a_raw =
        mlpl_runtime::call_builtin("randn", vec![DenseArray::from_scalar(a_seed), shape_arr])
            .map_err(|e| TuneError::RuntimeMessage(e.to_string()))?;
    let scale = 1.0 / (in_dim as f64).sqrt();
    let a_data: Vec<f64> = a_raw.data().iter().map(|v| v * scale).collect();
    Ok(DenseArray::new(Shape::new(vec![in_dim, rank]), a_data)?)
}
