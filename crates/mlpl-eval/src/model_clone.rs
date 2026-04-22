//! Saga 20 step 001: `clone_model(m) -> Model` builtin.
//!
//! Deep-copies a `ModelSpec` tree, allocating a fresh set of
//! parameter identifiers so the caller can mutate the copy (via
//! `perturb_params`, `adam`, etc.) without touching the source.
//! The cloned params inherit the source's tensor values and device
//! placement, so `apply(clone, X)` is forward-identical to
//! `apply(source, X)` until the caller perturbs the clone.
//!
//! See `contracts/eval-contract/clone-model.md` for the
//! behavioural contract.

use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::ModelSpec;
use crate::value::Value;

/// `clone_model(m)` -- deep-copies a model's spec and params.
///
/// The single argument must evaluate to a `Value::Model`. Each
/// layer in the source tree allocates a fresh layer id via
/// `env.next_model_id` (the same counter used by `linear`,
/// `embed`, and `attention` at construction time), producing
/// names like `__linear_W_{new_id}` that cannot collide with any
/// existing binding. Parameter values are copied by `DenseArray`
/// clone; device tags from `env.tensor_device(...)` propagate to
/// each fresh name.
pub(crate) fn eval_clone_model(
    args: &[Expr],
    env: &mut Environment,
) -> Result<ModelSpec, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "clone_model".into(),
            expected: 1,
            got: args.len(),
        });
    }
    // Bare identifier -> check the model registry first (models live
    // in `env.models`, not `env.vars`, so an Ident lookup via
    // `eval_expr` would miss them). Any other expression is evaluated
    // and must yield a `Value::Model` (e.g. `clone_model(linear(...))`
    // or `clone_model(chain(...))`).
    let source = if let Expr::Ident(name, _) = &args[0] {
        match env.get_model(name) {
            Some(m) => m.clone(),
            None => {
                return Err(EvalError::Unsupported(format!(
                    "clone_model: '{name}' is not a model"
                )));
            }
        }
    } else {
        match crate::eval::eval_expr(&args[0], env, &mut None)? {
            Value::Model(m) => m,
            _ => {
                return Err(EvalError::Unsupported(
                    "clone_model: argument must evaluate to a model".into(),
                ));
            }
        }
    };
    clone_spec(&source, env)
}

/// Recursively clone a `ModelSpec` tree, allocating fresh param
/// names for every parameterised node and copying tensor values
/// plus device tags into the environment. Parameter-free nodes
/// (activation, rms_norm) and compositional nodes (chain,
/// residual) recurse structurally; parameterised nodes delegate
/// to small helpers that allocate fresh names and copy the
/// underlying tensors.
fn clone_spec(spec: &ModelSpec, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    match spec {
        ModelSpec::Linear { w, b } => clone_linear(env, w, b),
        ModelSpec::Chain(children) => {
            let mut out = Vec::with_capacity(children.len());
            for child in children {
                out.push(clone_spec(child, env)?);
            }
            Ok(ModelSpec::Chain(out))
        }
        ModelSpec::Activation(kind) => Ok(ModelSpec::Activation(*kind)),
        ModelSpec::Residual(inner) => Ok(ModelSpec::Residual(Box::new(clone_spec(inner, env)?))),
        ModelSpec::RmsNorm { dim } => Ok(ModelSpec::RmsNorm { dim: *dim }),
        ModelSpec::Embedding {
            table,
            vocab,
            d_model,
        } => clone_embedding(env, table, *vocab, *d_model),
        ModelSpec::Attention {
            wq,
            wk,
            wv,
            wo,
            d_model,
            heads,
            causal,
        } => clone_attention(env, [wq, wk, wv, wo], *d_model, *heads, *causal),
    }
}

fn clone_linear(env: &mut Environment, w: &str, b: &str) -> Result<ModelSpec, EvalError> {
    let id = env.next_model_id;
    env.next_model_id += 1;
    let new_w = format!("__linear_W_{id}");
    let new_b = format!("__linear_b_{id}");
    copy_param(env, w, &new_w)?;
    copy_param(env, b, &new_b)?;
    Ok(ModelSpec::Linear { w: new_w, b: new_b })
}

fn clone_embedding(
    env: &mut Environment,
    table: &str,
    vocab: usize,
    d_model: usize,
) -> Result<ModelSpec, EvalError> {
    let id = env.next_model_id;
    env.next_model_id += 1;
    let new_table = format!("__embed_E_{id}");
    copy_param(env, table, &new_table)?;
    Ok(ModelSpec::Embedding {
        table: new_table,
        vocab,
        d_model,
    })
}

fn clone_attention(
    env: &mut Environment,
    projections: [&str; 4],
    d_model: usize,
    heads: usize,
    causal: bool,
) -> Result<ModelSpec, EvalError> {
    let id = env.next_model_id;
    env.next_model_id += 1;
    let new_wq = format!("__attn_Wq_{id}");
    let new_wk = format!("__attn_Wk_{id}");
    let new_wv = format!("__attn_Wv_{id}");
    let new_wo = format!("__attn_Wo_{id}");
    let new_names = [&new_wq, &new_wk, &new_wv, &new_wo];
    for (old, new) in projections.iter().zip(new_names.iter()) {
        copy_param(env, old, new)?;
    }
    Ok(ModelSpec::Attention {
        wq: new_wq,
        wk: new_wk,
        wv: new_wv,
        wo: new_wo,
        d_model,
        heads,
        causal,
    })
}

/// Copy the tensor value and device tag from `old` to `new` in the
/// environment, marking the new binding as a trainable parameter.
fn copy_param(env: &mut Environment, old: &str, new: &str) -> Result<(), EvalError> {
    let value = env
        .get(old)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(old.into()))?;
    let device = env.tensor_device(old).to_string();
    env.set_param(new.to_string(), value);
    env.set_tensor_device(new.to_string(), device);
    Ok(())
}
