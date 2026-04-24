//! Saga 22 step 001: `estimate_train(...)` feasibility /
//! resource-estimation builtin.
//!
//! Pure-math estimator over `ModelSpec` + loop shape.
//! Returns a rank-1 `[5]` f64 array with
//! `[params, vram_bytes, disk_bytes, flops,
//! wall_seconds]`. Honest-approximate at ~2x accuracy
//! -- activation memory uses a 4x safety factor,
//! elementwise / softmax / layer-norm costs are
//! ignored, and wall-clock depends on a user-settable
//! device-throughput key (default 50 GFLOPS). See
//! `contracts/eval-contract/estimate.md`.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::ModelSpec;
use crate::value::Value;

/// Conservative laptop-CPU lower bound (50 GFLOPS).
/// Overridable via
/// `env.set_string("mlpl_device_throughput_gflops",
/// ...)`; Saga 22 step 002's `calibrate_device()`
/// writes this key automatically.
const DEFAULT_GFLOPS: f64 = 50.0;

/// Safety multiplier on the activation-memory
/// heuristic. Conservative; covers Q/K/V/softmax
/// temporaries + residuals.
const ACTIVATION_FACTOR: f64 = 4.0;

/// Default element size (f64 -- what MLPL uses today).
const DEFAULT_DTYPE_BYTES: f64 = 8.0;

/// Per-spec structural statistics collected in one
/// tree walk: total parameter count, trainable
/// parameter count (frozen params excluded), the
/// widest "hidden" dim seen anywhere in the tree, and
/// the count of parameterized nodes. Returned as a
/// struct so the orchestrator does the VRAM math in
/// one place with named fields.
struct Stats {
    params: f64,
    trainable: f64,
    hidden: f64,
    depth: f64,
}

/// `estimate_train(model, steps, batch, seq [,
/// dtype_bytes]) -> [5]`.
pub(crate) fn eval_estimate_train(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    if !(4..=5).contains(&args.len()) {
        return Err(EvalError::BadArity {
            func: "estimate_train".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let spec = resolve_model_spec(&args[0], env)?;
    let steps = eval_pos_scalar(&args[1], env, "steps")?;
    let batch = eval_pos_scalar(&args[2], env, "batch_size")?;
    let seq = eval_pos_scalar(&args[3], env, "seq_len")?;
    let dtype_bytes = if args.len() == 5 {
        eval_pos_scalar(&args[4], env, "dtype_bytes")?
    } else {
        DEFAULT_DTYPE_BYTES
    };

    let mut stats = Stats {
        params: 0.0,
        trainable: 0.0,
        hidden: 0.0,
        depth: 0.0,
    };
    for name in spec.params() {
        let Some(arr) = env.get(&name) else {
            continue;
        };
        let size = arr.shape().dims().iter().product::<usize>() as f64;
        stats.params += size;
        if !env.is_frozen(&name) {
            stats.trainable += size;
        }
    }
    accumulate_hidden_depth(&spec, env, &mut stats);
    if stats.params == 0.0 {
        return Err(EvalError::Unsupported(
            "estimate_train: model has no trainable parameters".into(),
        ));
    }

    // VRAM = (weights + grad + adam_moments) * dtype
    //      + activation_bytes.
    let weight_bytes = stats.params * dtype_bytes;
    let grad_bytes = stats.trainable * dtype_bytes;
    let adam_bytes = 2.0 * stats.trainable * dtype_bytes;
    let activation_bytes =
        batch * seq * stats.hidden * stats.depth * dtype_bytes * ACTIVATION_FACTOR;
    let vram = weight_bytes + grad_bytes + adam_bytes + activation_bytes;

    let disk = stats.params * dtype_bytes;

    let flops_per_step = walk_flops_per_step(&spec, env, batch, seq);
    let flops = flops_per_step * steps;

    let gflops = env
        .get_string("mlpl_device_throughput_gflops")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_GFLOPS);
    let wall = flops / (gflops * 1e9);

    DenseArray::new(
        Shape::new(vec![5]),
        vec![stats.params, vram, disk, flops, wall],
    )
    .map_err(Into::into)
}

fn resolve_model_spec(arg: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if let Expr::Ident(name, _) = arg
        && let Some(m) = env.get_model(name)
    {
        return Ok(m.clone());
    }
    match crate::eval::eval_expr(arg, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(
            "estimate_train: first argument must be a model".into(),
        )),
    }
}

fn eval_pos_scalar(arg: &Expr, env: &mut Environment, name: &str) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(arg, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "estimate_train: {name} must be a scalar, got rank {}",
            arr.rank()
        )));
    }
    let v = arr.data()[0];
    if !v.is_finite() || v <= 0.0 {
        return Err(EvalError::Unsupported(format!(
            "estimate_train: {name} must be positive, got {v}"
        )));
    }
    Ok(v)
}

/// Depth-first walk that updates `hidden` (widest dim
/// observed) and `depth` (count of parameterized nodes).
/// Parameter sums come from `ModelSpec::params()` in the
/// orchestrator; keeping this walk small means its only
/// job is the two dimensional statistics.
fn accumulate_hidden_depth(spec: &ModelSpec, env: &Environment, acc: &mut Stats) {
    match spec {
        ModelSpec::Linear { w, .. } => {
            let (i, o) = linear_dims(env, w);
            acc.hidden = acc.hidden.max(i.max(o));
            acc.depth += 1.0;
        }
        ModelSpec::Embedding { d_model, .. } | ModelSpec::Attention { d_model, .. } => {
            acc.hidden = acc.hidden.max(*d_model as f64);
            acc.depth += 1.0;
        }
        ModelSpec::LinearLora {
            in_dim, out_dim, ..
        } => {
            acc.hidden = acc.hidden.max((*in_dim).max(*out_dim) as f64);
            acc.depth += 1.0;
        }
        ModelSpec::Chain(children) => {
            for c in children {
                accumulate_hidden_depth(c, env, acc);
            }
        }
        ModelSpec::Residual(inner) => accumulate_hidden_depth(inner, env, acc),
        ModelSpec::Activation(_) | ModelSpec::RmsNorm { .. } => {}
    }
}

fn linear_dims(env: &Environment, w_name: &str) -> (f64, f64) {
    match env.get(w_name) {
        Some(arr) if arr.rank() == 2 => {
            let d = arr.shape().dims();
            (d[0] as f64, d[1] as f64)
        }
        _ => (0.0, 0.0),
    }
}

fn walk_flops_per_step(spec: &ModelSpec, env: &Environment, batch: f64, seq: f64) -> f64 {
    match spec {
        ModelSpec::Linear { w, .. } => {
            let (in_dim, out_dim) = linear_dims(env, w);
            2.0 * in_dim * out_dim * batch
        }
        ModelSpec::Embedding { vocab, d_model, .. } => {
            2.0 * batch * (*vocab as f64) * (*d_model as f64)
        }
        ModelSpec::Attention { d_model, .. } => {
            // Projections Q/K/V/O: 4 * 2 * d^2 * batch * seq.
            // Scores QK^T + AV: 2 * 2 * s^2 * d * batch.
            let d = *d_model as f64;
            8.0 * d * d * batch * seq + 4.0 * seq * seq * d * batch
        }
        ModelSpec::LinearLora {
            in_dim,
            out_dim,
            rank,
            ..
        } => {
            let i = *in_dim as f64;
            let o = *out_dim as f64;
            let r = *rank as f64;
            2.0 * i * o * batch + 2.0 * i * r * batch + 2.0 * r * o * batch
        }
        ModelSpec::Chain(children) => children
            .iter()
            .map(|c| walk_flops_per_step(c, env, batch, seq))
            .sum(),
        ModelSpec::Residual(inner) => walk_flops_per_step(inner, env, batch, seq),
        ModelSpec::Activation(_) | ModelSpec::RmsNorm { .. } => 0.0,
    }
}
