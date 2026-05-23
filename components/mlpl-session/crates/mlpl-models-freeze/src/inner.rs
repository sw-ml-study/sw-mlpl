//! The `freeze_inner` / `unfreeze_inner` forward passes,
//! generic over the env capability traits and the caller's
//! error type.

use mlpl_array::DenseArray;
use mlpl_env_traits::{HasFrozen, HasModels};
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;

use crate::error::FreezeError;
use crate::resolve::resolve_model;

/// `freeze(m)` -- mark every parameter of `m` as frozen.
pub fn freeze_inner<E, F, Err>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
    func: &str,
) -> Result<DenseArray, Err>
where
    E: HasModels + HasFrozen,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<FreezeError>,
{
    let spec = resolve_model(args, env, resolve, func)?;
    for name in spec.params() {
        env.mark_frozen(&name);
    }
    Ok(DenseArray::from_scalar(0.0))
}

/// `unfreeze(m)` -- remove every parameter of `m` from the
/// frozen set. Inverse of `freeze`; idempotent.
pub fn unfreeze_inner<E, F, Err>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
    func: &str,
) -> Result<DenseArray, Err>
where
    E: HasModels + HasFrozen,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<FreezeError>,
{
    let spec = resolve_model(args, env, resolve, func)?;
    for name in spec.params() {
        env.unmark_frozen(&name);
    }
    Ok(DenseArray::from_scalar(0.0))
}
