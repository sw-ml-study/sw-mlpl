//! Saga 20 step 002: `perturb_params(m, family, sigma, seed)`.
//!
//! Walks a model's parameter set, filters by family, and adds
//! `sigma * randn(seed, shape)` to each matching parameter in
//! place. The core of the Neural Thickets workflow.
//!
//! See `contracts/eval-contract/perturb-params.md` for the
//! behavioural contract.

use std::collections::HashSet;

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::ModelSpec;

const FAMILIES: &[&str] = &["all_layers", "attention_only", "mlp_only", "embed_and_head"];

/// `perturb_params(m, family, sigma, seed)` -- apply family-targeted
/// Gaussian noise to a model's parameters, in place. Returns a
/// scalar zero so the call can sit in statement or expression
/// position, mirroring `to_device`'s unit-return convention.
pub(crate) fn eval_perturb_params(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    if args.len() != 4 {
        return Err(EvalError::BadArity {
            func: "perturb_params".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let model_name = match &args[0] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "perturb_params: first argument must be a model identifier".into(),
            ));
        }
    };
    let spec = env.get_model(&model_name).cloned().ok_or_else(|| {
        EvalError::Unsupported(format!("perturb_params: '{model_name}' is not a model"))
    })?;
    let family = match &args[1] {
        Expr::StrLit(s, _) => s.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "perturb_params: family (second argument) must be a string literal".into(),
            ));
        }
    };
    if !FAMILIES.contains(&family.as_str()) {
        return Err(EvalError::Unsupported(format!(
            "perturb_params: unknown family '{family}' (expected one of {})",
            FAMILIES.join(", ")
        )));
    }
    let sigma = scalar_f64(&args[2], env)?;
    let seed = scalar_f64(&args[3], env)?;

    let head = head_param_names(&spec);
    let affected = filter_family(&spec.params(), &family, &head);
    for (i, name) in affected.iter().enumerate() {
        perturb_one(env, name, sigma, seed + i as f64)?;
    }
    Ok(DenseArray::from_scalar(0.0))
}

/// Parameter names owned by the "final projection head": the last
/// top-level `Linear` child of the outermost `Chain`, or the model
/// itself if it is a bare `Linear`. Returns an empty set when the
/// outermost node is anything else (e.g. a `Residual` wrapper), in
/// which case no param is treated as the head.
fn head_param_names(spec: &ModelSpec) -> HashSet<String> {
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

/// Filter an ordered list of param names by family membership. The
/// head set is consulted so `mlp_only` can exclude the projection
/// head and `embed_and_head` can include it.
fn filter_family(all: &[String], family: &str, head: &HashSet<String>) -> Vec<String> {
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

/// Apply `sigma * randn(seed, shape(param))` to the named parameter
/// in place. Preserves `env.params` membership and device tag
/// because we only touch the underlying value via `env.set`.
fn perturb_one(env: &mut Environment, name: &str, sigma: f64, seed: f64) -> Result<(), EvalError> {
    let old = env
        .get(name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(name.into()))?;
    let shape_dims: Vec<f64> = old.shape().dims().iter().map(|&d| d as f64).collect();
    let shape_arr = DenseArray::new(Shape::new(vec![shape_dims.len()]), shape_dims)?;
    let noise =
        mlpl_runtime::call_builtin("randn", vec![DenseArray::from_scalar(seed), shape_arr])?;
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

/// Evaluate `expr` and require that it is a rank-0 scalar. Local
/// clone of the private helper in `model_dispatch.rs`; duplicated
/// rather than promoted to `pub(crate)` to avoid touching an
/// already-over-budget module.
fn scalar_f64(expr: &Expr, env: &mut Environment) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(
            "perturb_params: sigma and seed must be scalars".into(),
        ));
    }
    Ok(arr.data()[0])
}
