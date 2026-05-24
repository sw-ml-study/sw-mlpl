//! `perturb_params(m, family, sigma, seed)`: walks a model's
//! parameter set, filters by family, and adds
//! `sigma * randn(seed, shape)` to each matching parameter in
//! place. Per-name + family-filter helpers live in
//! `perturb_helpers`.

use mlpl_array::DenseArray;
use mlpl_env_traits::{HasModels, HasVars};
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;

use crate::error::MutateError;
use crate::perturb_helpers::{filter_family, head_param_names, perturb_one};

const FAMILIES: &[&str] = &["all_layers", "attention_only", "mlp_only", "embed_and_head"];

pub fn perturb_params_inner<E, F, Err>(
    args: &[Expr],
    env: &mut E,
    scalar: F,
) -> Result<DenseArray, Err>
where
    E: HasModels + HasVars,
    F: Fn(&Expr, &mut E) -> Result<f64, Err>,
    Err: From<MutateError>,
{
    let (spec, family, sigma, seed) = parse_args(args, env, scalar)?;
    let head = head_param_names(&spec);
    let affected = filter_family(&spec.params(), &family, &head);
    for (i, name) in affected.iter().enumerate() {
        perturb_one(env, name, sigma, seed + i as f64)?;
    }
    Ok(DenseArray::from_scalar(0.0))
}

fn parse_args<E, F, Err>(
    args: &[Expr],
    env: &mut E,
    scalar: F,
) -> Result<(ModelSpec, String, f64, f64), Err>
where
    E: HasModels,
    F: Fn(&Expr, &mut E) -> Result<f64, Err>,
    Err: From<MutateError>,
{
    let [a0, a1, a2, a3] = args else {
        return Err(MutateError::BadArity {
            func: "perturb_params".into(),
            expected: 4,
            got: args.len(),
        }
        .into());
    };
    let spec = resolve_model_ident(a0, env)?;
    let family = resolve_family(a1)?;
    let sigma = scalar(a2, env)?;
    let seed = scalar(a3, env)?;
    Ok((spec, family, sigma, seed))
}

fn resolve_model_ident<E: HasModels, Err: From<MutateError>>(
    arg: &Expr,
    env: &mut E,
) -> Result<ModelSpec, Err> {
    let Expr::Ident(name, _) = arg else {
        return Err(MutateError::NotAModelExpr("perturb_params".into()).into());
    };
    env.get_model(name).cloned().ok_or_else(|| {
        MutateError::NotAModel {
            func: "perturb_params".into(),
            name: name.clone(),
        }
        .into()
    })
}

fn resolve_family<Err: From<MutateError>>(arg: &Expr) -> Result<String, Err> {
    let Expr::StrLit(s, _) = arg else {
        return Err(MutateError::ExpectedString("perturb_params".into()).into());
    };
    if !FAMILIES.contains(&s.as_str()) {
        return Err(MutateError::UnknownFamily {
            family: s.clone(),
            valid: FAMILIES.iter().map(|s| (*s).to_string()).collect(),
        }
        .into());
    }
    Ok(s.clone())
}
