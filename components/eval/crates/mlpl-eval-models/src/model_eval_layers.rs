//! Saga 33 step 004: parameter-allocating layer constructors
//! (`linear`, `embed`, `rms_norm`). Each builder mints a fresh
//! `ModelSpec` and stamps the freshly allocated tensor params
//! into `env` under generated names, recording device placement
//! and a `ValueTag::Weight` for traceability.

use crate::env_api::{EnvDevice, EnvParams, EnvTags, EnvTensorDevice};
use mlpl_array::{DenseArray, Shape};
use mlpl_core::ValueTag;
use mlpl_parser::Expr;

use crate::model_dispatch_scalar::{scalar_f64, scalar_usize};
use crate::model_layer_options::numeric_options;
use mlpl_eval_core::model::{DEFAULT_RMS_EPS, ModelSpec};
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// `linear(in_dim, out_dim, seed[, {bias: 0}])`.
pub fn eval_linear(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if !(3..=4).contains(&args.len()) {
        return Err(EvalError::BadArity {
            func: "linear".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let in_dim = scalar_usize(&args[0], env, "linear")?;
    let out_dim = scalar_usize(&args[1], env, "linear")?;
    let seed = scalar_f64(&args[2], env, "linear")?;
    let [bias] = numeric_options("linear", args, 3, [("bias", 1.0)], env)?;
    init_linear_params(env, (in_dim, out_dim), seed, bias != 0.0)
}

/// `embed(vocab_size, d_model, seed)` -- token embedding layer.
pub fn eval_embedding(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "embed".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let vocab = scalar_usize(&args[0], env, "embed")?;
    let d_model = scalar_usize(&args[1], env, "embed")?;
    let seed = scalar_f64(&args[2], env, "embed")?;

    let id = env.next_model_id;
    env.next_model_id += 1;
    let table_name = format!("__embed_E_{id}");
    let layer = format!("embed_{id}");

    let table_init = mlpl_runtime::call_builtin(
        "randn",
        vec![
            DenseArray::from_scalar(seed),
            DenseArray::new(Shape::new(vec![2]), vec![vocab as f64, d_model as f64])?,
        ],
    )?;
    let table_data: Vec<f64> = table_init.data().iter().map(|v| v * 0.1).collect();
    let table = DenseArray::new(Shape::new(vec![vocab, d_model]), table_data)?;
    let device = env.device().to_string();
    env.set_param(table_name.clone(), table);
    env.set_tensor_device(table_name.clone(), device);
    env.set_tag(
        table_name.clone(),
        ValueTag::Weight {
            layer,
            name: "table".into(),
        },
    );

    Ok(ModelSpec::Embedding {
        table: table_name,
        vocab,
        d_model,
    })
}

/// `rms_norm(dim[, {eps}])` -- parameter-free RMS normalization over the
/// last axis.
pub fn eval_rms_norm(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if !(1..=2).contains(&args.len()) {
        return Err(EvalError::BadArity {
            func: "rms_norm".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let dim = scalar_usize(&args[0], env, "rms_norm")?;
    let [eps] = numeric_options("rms_norm", args, 1, [("eps", DEFAULT_RMS_EPS)], env)?;
    Ok(ModelSpec::RmsNorm { dim, eps })
}

/// Mint a linear layer: its generated names, weight (scaled randn) and,
/// unless bias-free, bias (zeros) -- param + device stamp + Weight/Bias
/// tags.
fn init_linear_params(
    env: &mut Environment,
    dims: (usize, usize),
    seed: f64,
    with_bias: bool,
) -> Result<ModelSpec, EvalError> {
    let id = env.next_model_id;
    env.next_model_id += 1;
    let (w_name, layer) = (format!("__linear_W_{id}"), format!("linear_{id}"));
    let spec = ModelSpec::Linear {
        w: w_name.clone(),
        b: with_bias.then(|| format!("__linear_b_{id}")),
    };
    let (in_dim, out_dim) = dims;
    let w_init = mlpl_runtime::call_builtin(
        "randn",
        vec![
            DenseArray::from_scalar(seed),
            DenseArray::new(Shape::new(vec![2]), vec![in_dim as f64, out_dim as f64])?,
        ],
    )?;
    let w_data: Vec<f64> = w_init.data().iter().map(|v| v * 0.5).collect();
    let w = DenseArray::new(Shape::new(vec![in_dim, out_dim]), w_data)?;
    let device = env.device().to_string();
    env.set_param(w_name.clone(), w);
    env.set_tensor_device(w_name.clone(), device.clone());
    let tag = ValueTag::Weight {
        layer: layer.clone(),
        name: "W".into(),
    };
    env.set_tag(w_name, tag);
    if let ModelSpec::Linear {
        b: Some(b_name), ..
    } = &spec
    {
        let b = DenseArray::zeros(Shape::new(vec![1, out_dim]));
        env.set_param(b_name.clone(), b);
        env.set_tensor_device(b_name.clone(), device);
        env.set_tag(b_name.clone(), ValueTag::Bias { layer });
    }
    Ok(spec)
}
