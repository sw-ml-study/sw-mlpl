//! `estimate_hypothetical(name, steps, batch, seq[, dtype,
//! lora_rank])` -- pure-math cost estimate for a hypothetical
//! HF-scale transformer named in the hardcoded spec table.
//! No model weights required; consults a static lookup.

use mlpl_array::{DenseArray, Shape};
use mlpl_env_traits::HasStrings;
use mlpl_parser::Expr;

use crate::error::FeasibilityError;
use crate::hypothetical_specs::{HypSpec, lookup_hypothetical};

const DEFAULT_GFLOPS: f64 = 50.0;
const ACTIVATION_FACTOR: f64 = 4.0;

pub fn estimate_hypothetical_inner<E, M, S, Err>(
    args: &[Expr],
    env: &mut E,
    name_resolver: M,
    scalar: S,
) -> Result<DenseArray, Err>
where
    E: HasStrings,
    M: FnOnce(&Expr, &mut E) -> Result<String, Err>,
    S: Fn(&Expr, &mut E) -> Result<f64, Err>,
    Err: From<FeasibilityError>,
{
    if !(4..=6).contains(&args.len()) {
        return Err(FeasibilityError::BadArity {
            func: "estimate_hypothetical".into(),
            expected: 4,
            got: args.len(),
        }
        .into());
    }
    let name = name_resolver(&args[0], env)?;
    let steps = pos_scalar(&args[1], env, "steps", &scalar)?;
    let batch = pos_scalar(&args[2], env, "batch_size", &scalar)?;
    let seq = pos_scalar(&args[3], env, "seq_len", &scalar)?;
    let dtype_bytes = if args.len() >= 5 {
        pos_scalar(&args[4], env, "dtype_bytes", &scalar)?
    } else {
        8.0
    };
    let lora_rank = if args.len() == 6 {
        pos_scalar(&args[5], env, "lora_rank", &scalar)?
    } else {
        0.0
    };
    let spec = lookup_hypothetical(&name)?;
    let gflops = device_gflops(env);
    let data = compute_hyp_estimate(&spec, steps, batch, seq, dtype_bytes, lora_rank, gflops);
    Ok(DenseArray::new(Shape::new(vec![5]), data).map_err(FeasibilityError::ArrayError)?)
}

fn pos_scalar<E, S, Err>(arg: &Expr, env: &mut E, name: &str, scalar: &S) -> Result<f64, Err>
where
    S: Fn(&Expr, &mut E) -> Result<f64, Err>,
    Err: From<FeasibilityError>,
{
    let v = scalar(arg, env)?;
    if !v.is_finite() || v <= 0.0 {
        return Err(FeasibilityError::NotPositive {
            func: "estimate_hypothetical".into(),
            name: name.into(),
            value: v,
        }
        .into());
    }
    Ok(v)
}

fn device_gflops<E: HasStrings>(env: &E) -> f64 {
    env.get_string("mlpl_device_throughput_gflops")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_GFLOPS)
}

/// Compute `[params, vram, disk, flops, wall]` from the
/// hypothetical spec. Pure arithmetic; takes no env.
fn compute_hyp_estimate(
    s: &HypSpec,
    steps: f64,
    batch: f64,
    seq: f64,
    dtype: f64,
    lora_rank: f64,
    gflops: f64,
) -> Vec<f64> {
    let per_layer_params = 4.0 * s.d_model * s.d_model + 2.0 * s.d_model * s.intermediate;
    let params = s.vocab * s.d_model + s.layers * per_layer_params + s.d_model * s.vocab;
    let trainable = if lora_rank > 0.0 {
        let per_layer_adapters = 4.0 * (s.d_model + s.d_model) * lora_rank
            + 2.0 * (s.d_model + s.intermediate) * lora_rank;
        s.layers * per_layer_adapters + (s.d_model + s.vocab) * lora_rank
    } else {
        params
    };
    let depth = 2.0 + 2.0 * s.layers;
    let vram = (params + trainable + 2.0 * trainable) * dtype
        + batch * seq * s.d_model * depth * dtype * ACTIVATION_FACTOR;
    let disk = params * dtype;
    let per_step = 2.0 * batch * s.vocab * s.d_model
        + s.layers
            * (8.0 * s.d_model * s.d_model * batch * seq
                + 4.0 * seq * seq * s.d_model * batch
                + 4.0 * s.d_model * s.intermediate * batch * seq)
        + 2.0 * batch * s.d_model * s.vocab;
    let flops = per_step * steps;
    let wall = flops / (gflops * 1e9);
    vec![params, vram, disk, flops, wall]
}
