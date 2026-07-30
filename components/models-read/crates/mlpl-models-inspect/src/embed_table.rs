//! `embed_table(model) -> [vocab, d_model]`. Walks the model
//! tree depth-first left-to-right and returns the first
//! Embedding layer's lookup table cloned from env.

use mlpl_array::DenseArray;
use mlpl_env_traits::{HasModels, HasVars};
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;

use crate::error::InspectError;

pub fn embed_table_inner<E, F, Err>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
) -> Result<DenseArray, Err>
where
    E: HasModels + HasVars,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<InspectError>,
{
    let [arg] = args else {
        return Err(InspectError::BadArity {
            func: "embed_table".into(),
            expected: 1,
            got: args.len(),
        }
        .into());
    };
    let spec = resolve_source(arg, env, resolve)?;
    match find_embedding_table(&spec, env) {
        Some(table) => Ok(table),
        None => Err(InspectError::NoEmbedding.into()),
    }
}

fn resolve_source<E, F, Err>(arg: &Expr, env: &mut E, resolve: F) -> Result<ModelSpec, Err>
where
    E: HasModels,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<InspectError>,
{
    if let Expr::Ident(name, _) = arg {
        return env.get_model(name).cloned().ok_or_else(|| {
            InspectError::NotAModel {
                func: "embed_table".into(),
                name: name.clone(),
            }
            .into()
        });
    }
    resolve(arg, env)
}

/// Depth-first left-to-right walk. Returns the first Embedding
/// layer's table as a fresh `DenseArray`, or `None` if the
/// subtree has no Embedding node.
fn find_embedding_table<E: HasVars>(spec: &ModelSpec, env: &E) -> Option<DenseArray> {
    match spec {
        ModelSpec::Embedding { table, .. } => env.get(table).cloned(),
        ModelSpec::Chain(children) => {
            for child in children {
                if let Some(t) = find_embedding_table(child, env) {
                    return Some(t);
                }
            }
            None
        }
        ModelSpec::Residual(inner) => find_embedding_table(inner, env),
        ModelSpec::Linear { .. }
        | ModelSpec::Activation(_)
        | ModelSpec::RmsNorm { .. }
        | ModelSpec::Attention { .. }
        | ModelSpec::LinearLora { .. }
        | ModelSpec::Engram { .. } => None,
    }
}
