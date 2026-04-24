//! Saga 16.5 step 002: `embed_table(model)` builtin.
//!
//! Walks a `ModelSpec` tree and returns the first
//! Embedding layer's `[vocab, d_model]` table. Closes
//! the Saga 16 gap where a trained
//! `chain(embed, transformer_block, head)` had no
//! source-level way to pull the learned embedding
//! back out.
//!
//! First-match semantics: if a model somehow has two
//! Embedding layers (not a shipped pattern), the one
//! returned is the first encountered in depth-first
//! left-to-right order.
//!
//! See `contracts/eval-contract/embed-table.md`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::ModelSpec;
use crate::value::Value;

/// `embed_table(model) -> [vocab, d_model]`. Resolves the
/// argument to a `ModelSpec`, walks the tree depth-first,
/// and returns the first Embedding layer's lookup table
/// cloned out of `env`.
pub(crate) fn eval_embed_table(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "embed_table".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let spec = if let Expr::Ident(name, _) = &args[0] {
        match env.get_model(name) {
            Some(m) => m.clone(),
            None => {
                return Err(EvalError::Unsupported(format!(
                    "embed_table: '{name}' is not a model"
                )));
            }
        }
    } else {
        match crate::eval::eval_expr(&args[0], env, &mut None)? {
            Value::Model(m) => m,
            _ => {
                return Err(EvalError::Unsupported(
                    "embed_table: argument must evaluate to a model".into(),
                ));
            }
        }
    };
    match find_embedding_table(&spec, env) {
        Some(table) => Ok(table),
        None => Err(EvalError::Unsupported(
            "embed_table: model contains no Embedding layer".into(),
        )),
    }
}

/// Depth-first left-to-right walk. Returns the first
/// Embedding layer's table as a fresh `DenseArray`, or
/// `None` if the subtree has no Embedding node.
fn find_embedding_table(spec: &ModelSpec, env: &Environment) -> Option<DenseArray> {
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
        | ModelSpec::LinearLora { .. } => None,
    }
}
