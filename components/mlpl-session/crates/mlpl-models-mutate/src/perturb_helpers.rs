//! Per-name perturbation + family-filter helpers extracted
//! from `perturb.rs` so each module stays under the 4-fn warn
//! line.

use std::collections::HashSet;

use mlpl_array::{DenseArray, Shape};
use mlpl_env_traits::HasVars;
use mlpl_eval_core::model::ModelSpec;

use crate::error::MutateError;

/// Parameter names owned by the "final projection head": the
/// last top-level Linear in the outermost Chain, or the model
/// itself if it is a bare Linear.
pub(crate) fn head_param_names(spec: &ModelSpec) -> HashSet<String> {
    let head_linear = match spec {
        ModelSpec::Linear { .. } => Some(spec),
        ModelSpec::Chain(children) => children
            .iter()
            .rev()
            .find(|c| matches!(c, ModelSpec::Linear { .. })),
        _ => None,
    };
    match head_linear {
        Some(ModelSpec::Linear { w, b }) => [w.clone(), b.clone()].into_iter().collect(),
        _ => HashSet::new(),
    }
}

/// Filter an ordered list of param names by family membership.
pub(crate) fn filter_family(all: &[String], family: &str, head: &HashSet<String>) -> Vec<String> {
    all.iter()
        .filter(|name| match family {
            "all_layers" => true,
            "attention_only" => name.starts_with("__attn_"),
            "mlp_only" => name.starts_with("__linear_") && !head.contains(*name),
            "embed_and_head" => name.starts_with("__embed_") || head.contains(*name),
            _ => false,
        })
        .cloned()
        .collect()
}

/// Apply `sigma * randn(seed, shape(param))` to the named
/// parameter in place via `HasVars::set`.
pub(crate) fn perturb_one<E: HasVars>(
    env: &mut E,
    name: &str,
    sigma: f64,
    seed: f64,
) -> Result<(), MutateError> {
    let old = env
        .get(name)
        .cloned()
        .ok_or_else(|| MutateError::UndefinedVariable(name.into()))?;
    let noise = randn_like(&old, seed)?;
    let new_data: Vec<f64> = old
        .data()
        .iter()
        .zip(noise.data().iter())
        .map(|(o, n)| o + sigma * n)
        .collect();
    let new_tensor = DenseArray::new(old.shape().clone(), new_data)?;
    env.set(name.to_string(), new_tensor);
    Ok(())
}

fn randn_like(template: &DenseArray, seed: f64) -> Result<DenseArray, MutateError> {
    let shape_dims: Vec<f64> = template.shape().dims().iter().map(|&d| d as f64).collect();
    let shape_arr = DenseArray::new(Shape::new(vec![shape_dims.len()]), shape_dims)?;
    mlpl_runtime::call_builtin("randn", vec![DenseArray::from_scalar(seed), shape_arr])
        .map_err(|e| MutateError::RuntimeMessage(e.to_string()))
}
