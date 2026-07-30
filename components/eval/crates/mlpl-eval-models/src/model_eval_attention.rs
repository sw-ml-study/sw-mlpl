//! Saga 33 step 004: multi-head attention layer constructor.
//! `attention(d_model, heads, seed)` and `causal_attention(...)`
//! dispatch through the same builder, differing only in the
//! `causal` flag passed to `ModelSpec::Attention`.

use crate::env_api::{EnvDevice, EnvParams, EnvTags, EnvTensorDevice};
use mlpl_array::{DenseArray, Shape};
use mlpl_core::ValueTag;
use mlpl_parser::Expr;

use crate::model_dispatch_scalar::{scalar_f64, scalar_usize};
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

pub fn eval_attention(
    args: &[Expr],
    env: &mut Environment,
    causal: bool,
) -> Result<ModelSpec, EvalError> {
    let func = if causal {
        "causal_attention"
    } else {
        "attention"
    };
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: func.into(),
            expected: 3,
            got: args.len(),
        });
    }
    let d_model = scalar_usize(&args[0], env, func)?;
    let heads = scalar_usize(&args[1], env, func)?;
    let seed = scalar_f64(&args[2], env, func)?;
    if heads == 0 || d_model % heads != 0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: d_model ({d_model}) must be divisible by heads ({heads})"
        )));
    }
    let id = env.next_model_id;
    env.next_model_id += 1;
    let wq = format!("__attn_Wq_{id}");
    let wk = format!("__attn_Wk_{id}");
    let wv = format!("__attn_Wv_{id}");
    let wo = format!("__attn_Wo_{id}");
    let layer = format!("attention_{id}");
    init_projection_params(env, [&wq, &wk, &wv, &wo], d_model, seed, &layer)?;
    Ok(ModelSpec::Attention {
        wq,
        wk,
        wv,
        wo,
        d_model,
        heads,
        causal,
    })
}

/// Initialize the four projection weight matrices (`W_q`/`W_k`/`W_v`/
/// `W_o`): scaled-randn init, param + device stamp + Weight tag.
fn init_projection_params(
    env: &mut Environment,
    names: [&String; 4],
    d_model: usize,
    seed: f64,
    layer: &str,
) -> Result<(), EvalError> {
    let device = env.device().to_string();
    let proj_names = ["W_q", "W_k", "W_v", "W_o"];
    for (i, name) in names.iter().enumerate() {
        let init = mlpl_runtime::call_builtin(
            "randn",
            vec![
                DenseArray::from_scalar(seed + i as f64),
                DenseArray::new(Shape::new(vec![2]), vec![d_model as f64, d_model as f64])?,
            ],
        )?;
        let scaled: Vec<f64> = init.data().iter().map(|v| v * 0.5).collect();
        let arr = DenseArray::new(Shape::new(vec![d_model, d_model]), scaled)?;
        env.set_param((*name).clone(), arr);
        env.set_tensor_device((*name).clone(), device.clone());
        env.set_tag(
            (*name).clone(),
            ValueTag::Weight {
                layer: layer.to_string(),
                name: proj_names[i].into(),
            },
        );
    }
    Ok(())
}
