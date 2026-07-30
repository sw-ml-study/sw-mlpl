//! `clone_model(m) -> Model`: deep-copy a `ModelSpec` tree,
//! allocating fresh parameter identifiers so the caller can
//! mutate the copy without touching the source. Generic over
//! the env capability traits the per-variant helpers need.

use mlpl_env_traits::{HasModelIds, HasModels, HasParams, HasTensorDevices, HasVars};
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;

use crate::clone_attention::{clone_attention, clone_linear_lora};
use crate::clone_variants::{clone_chain, clone_embedding, clone_linear};
use crate::error::MutateError;

/// `clone_model(m)` -- deep-copies a model's spec and params.
pub fn clone_model_inner<E, F, Err>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
) -> Result<ModelSpec, Err>
where
    E: HasModels + HasVars + HasParams + HasTensorDevices + HasModelIds,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<MutateError>,
{
    let [arg] = args else {
        return Err(MutateError::BadArity {
            func: "clone_model".into(),
            expected: 1,
            got: args.len(),
        }
        .into());
    };
    let source = resolve_source(arg, env, resolve)?;
    clone_spec(&source, env).map_err(Into::into)
}

fn resolve_source<E, F, Err>(arg: &Expr, env: &mut E, resolve: F) -> Result<ModelSpec, Err>
where
    E: HasModels,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<MutateError>,
{
    if let Expr::Ident(name, _) = arg {
        return env.get_model(name).cloned().ok_or_else(|| {
            MutateError::NotAModel {
                func: "clone_model".into(),
                name: name.clone(),
            }
            .into()
        });
    }
    resolve(arg, env)
}

/// Recursively clone a `ModelSpec` tree. Public so callers
/// (e.g. mlpl-eval's `model_lora`) can use it independently of
/// the `clone_model_inner` entry point.
pub fn clone_spec<E>(spec: &ModelSpec, env: &mut E) -> Result<ModelSpec, MutateError>
where
    E: HasVars + HasParams + HasTensorDevices + HasModelIds,
{
    match spec {
        ModelSpec::Linear { w, b } => clone_linear(env, w, b),
        ModelSpec::Chain(children) => clone_chain(children, env),
        ModelSpec::Activation(kind) => Ok(ModelSpec::Activation(*kind)),
        ModelSpec::Residual(inner) => Ok(ModelSpec::Residual(Box::new(clone_spec(inner, env)?))),
        ModelSpec::RmsNorm { dim } => Ok(ModelSpec::RmsNorm { dim: *dim }),
        ModelSpec::Embedding {
            table,
            vocab,
            d_model,
        } => clone_embedding(env, table, *vocab, *d_model),
        ModelSpec::Attention { .. } => clone_attention(spec, env),
        ModelSpec::LinearLora { .. } => clone_linear_lora(spec, env),
        ModelSpec::Engram { .. } => Err(MutateError::RuntimeMessage(
            "clone_model: engram layers are not cloneable yet (planned with the \
             engram training saga)"
                .into(),
        )),
    }
}
