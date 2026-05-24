//! `estimate_train(model, steps, batch, seq [, dtype_bytes])
//! -> [5]`. Pure-math estimator over a `ModelSpec` + loop
//! shape: `[params, vram_bytes, disk_bytes, flops,
//! wall_seconds]`. Honest-approximate at ~2x accuracy.

use mlpl_array::{DenseArray, Shape};
use mlpl_env_traits::{HasFrozen, HasModels, HasStrings, HasVars};
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;

use crate::error::InspectError;
use crate::estimate_compute::{DEFAULT_DTYPE_BYTES, collect_stats, compute_vram, device_gflops};
use crate::estimate_walk::walk_flops_per_step;

pub fn estimate_train_inner<E, M, S, Err>(
    args: &[Expr],
    env: &mut E,
    resolve_model: M,
    scalar: S,
) -> Result<DenseArray, Err>
where
    E: HasModels + HasVars + HasFrozen + HasStrings,
    M: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    S: Fn(&Expr, &mut E, &str) -> Result<f64, Err>,
    Err: From<InspectError>,
{
    let (spec, steps, batch, seq, dtype_bytes) = parse_args(args, env, resolve_model, scalar)?;
    let stats = collect_stats(&spec, env)?;
    let vram = compute_vram(&stats, batch, seq, dtype_bytes);
    let disk = stats.params * dtype_bytes;
    let flops = walk_flops_per_step(&spec, env, batch, seq) * steps;
    let wall = flops / (device_gflops(env) * 1e9);
    DenseArray::new(
        Shape::new(vec![5]),
        vec![stats.params, vram, disk, flops, wall],
    )
    .map_err(|e| InspectError::ArrayError(e).into())
}

fn parse_args<E, M, S, Err>(
    args: &[Expr],
    env: &mut E,
    resolve_model: M,
    scalar: S,
) -> Result<(ModelSpec, f64, f64, f64, f64), Err>
where
    E: HasModels,
    M: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    S: Fn(&Expr, &mut E, &str) -> Result<f64, Err>,
    Err: From<InspectError>,
{
    if !(4..=5).contains(&args.len()) {
        return Err(InspectError::BadArity {
            func: "estimate_train".into(),
            expected: 4,
            got: args.len(),
        }
        .into());
    }
    let spec = resolve_model_arg(&args[0], env, resolve_model)?;
    let steps = scalar(&args[1], env, "steps")?;
    let batch = scalar(&args[2], env, "batch_size")?;
    let seq = scalar(&args[3], env, "seq_len")?;
    let dtype_bytes = if args.len() == 5 {
        scalar(&args[4], env, "dtype_bytes")?
    } else {
        DEFAULT_DTYPE_BYTES
    };
    Ok((spec, steps, batch, seq, dtype_bytes))
}

fn resolve_model_arg<E, M, Err>(arg: &Expr, env: &mut E, resolve: M) -> Result<ModelSpec, Err>
where
    E: HasModels,
    M: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<InspectError>,
{
    if let Expr::Ident(name, _) = arg
        && let Some(m) = env.get_model(name)
    {
        return Ok(m.clone());
    }
    resolve(arg, env)
}
