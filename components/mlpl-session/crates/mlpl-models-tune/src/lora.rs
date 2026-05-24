//! `lora(m, rank, alpha, seed)` -- wrap every `Linear` in
//! `m` with trainable low-rank adapters. Generic over the
//! env capability traits + caller-supplied resolver closures
//! so this crate never imports mlpl-eval.

use mlpl_env_traits::{HasFrozen, HasModelIds, HasModels, HasParams, HasTensorDevices, HasVars};
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;

use crate::error::TuneError;
use crate::lora_helpers::{LoraCtx, rewrite_linears_to_lora};

/// `lora(m, rank, alpha, seed)` entry point.
pub fn lora_inner<E, M, S, Err>(
    args: &[Expr],
    env: &mut E,
    resolve_model: M,
    scalar: S,
) -> Result<ModelSpec, Err>
where
    E: HasModels + HasVars + HasParams + HasFrozen + HasTensorDevices + HasModelIds,
    M: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    S: Fn(&Expr, &mut E) -> Result<f64, Err>,
    Err: From<TuneError>,
{
    let [a0, a1, a2, a3] = args else {
        return Err(TuneError::BadArity {
            func: "lora".into(),
            expected: 4,
            got: args.len(),
        }
        .into());
    };
    let source = resolve_source(a0, env, resolve_model)?;
    reject_nested_lora(&source)?;
    let rank = parse_rank(scalar(a1, env)?)?;
    let alpha = scalar(a2, env)?;
    let seed = scalar(a3, env)?;
    let cloned = mlpl_models_mutate::clone_spec(&source, env).map_err(TuneError::from)?;
    let mut ctx = LoraCtx {
        rank,
        alpha,
        seed,
        counter: 0,
    };
    let student = rewrite_linears_to_lora(cloned, env, &mut ctx).map_err(Err::from)?;
    auto_freeze_base(&student, env);
    Ok(student)
}

fn resolve_source<E, M, Err>(arg: &Expr, env: &mut E, resolve: M) -> Result<ModelSpec, Err>
where
    E: HasModels,
    M: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<TuneError>,
{
    if let Expr::Ident(name, _) = arg {
        return env
            .get_model(name)
            .cloned()
            .ok_or_else(|| TuneError::NotAModel(name.clone()).into());
    }
    resolve(arg, env)
}

fn reject_nested_lora<Err: From<TuneError>>(spec: &ModelSpec) -> Result<(), Err> {
    match spec {
        ModelSpec::LinearLora { .. } => Err(TuneError::NestedLora.into()),
        ModelSpec::Chain(children) => {
            for c in children {
                reject_nested_lora::<Err>(c)?;
            }
            Ok(())
        }
        ModelSpec::Residual(inner) => reject_nested_lora::<Err>(inner),
        _ => Ok(()),
    }
}

fn parse_rank<Err: From<TuneError>>(rank_f: f64) -> Result<usize, Err> {
    if !rank_f.is_finite() || rank_f < 0.0 || rank_f.fract() != 0.0 {
        return Err(TuneError::BadRank(rank_f).into());
    }
    let rank = rank_f as usize;
    if rank == 0 {
        return Err(TuneError::ZeroRank.into());
    }
    Ok(rank)
}

fn auto_freeze_base<E: HasFrozen>(student: &ModelSpec, env: &mut E) {
    for name in student.params() {
        if !name.starts_with("__lora_A_") && !name.starts_with("__lora_B_") {
            env.mark_frozen(&name);
        }
    }
}
