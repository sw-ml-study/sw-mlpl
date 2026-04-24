//! Saga 22 step 002: `calibrate_device` +
//! `estimate_hypothetical` + `feasible` builtins.
//!
//! The feasibility toolkit layered on top of
//! `estimate_train`. `calibrate_device(size?)`
//! benchmarks matmul on the active device and caches
//! the observed GFLOPS into
//! `env.mlpl_device_throughput_gflops` so subsequent
//! `estimate_train` calls produce honest wall-clock
//! numbers. `estimate_hypothetical(name, ...)` answers
//! "how big would a SmolLM / Llama / Qwen fine-tune
//! be on my laptop?" WITHOUT materializing any
//! weights, by consulting a hardcoded spec table.
//! `feasible(est, budget)` is the guard pattern used
//! to gate a real `train { }` call.
//!
//! See `contracts/eval-contract/calibrate-device.md`,
//! `contracts/eval-contract/estimate-hypothetical.md`,
//! and `contracts/eval-contract/feasible.md`.

use std::time::Instant;

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;

const DEFAULT_BENCH_SIZE: usize = 1024;
const BENCH_ITERS: u32 = 10;
const DEFAULT_GFLOPS: f64 = 50.0;
const ACTIVATION_FACTOR: f64 = 4.0;

/// Hardcoded structural spec for a hypothetical HF-
/// scale transformer. Dims come from each model's
/// published config; parameter-count totals line up
/// within ~30% of each model's reported size.
struct HypSpec {
    vocab: f64,
    d_model: f64,
    layers: f64,
    intermediate: f64,
}

/// `calibrate_device([size])` -- run a size x size
/// matmul `BENCH_ITERS` times on the active device,
/// return measured GFLOPS, cache result into
/// `env.mlpl_device_throughput_gflops`.
pub(crate) fn eval_calibrate_device(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    if args.len() > 1 {
        return Err(EvalError::BadArity {
            func: "calibrate_device".into(),
            expected: 0,
            got: args.len(),
        });
    }
    let size = if let Some(a) = args.first() {
        let arr = crate::eval::eval_expr(a, env, &mut None)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::Unsupported(
                "calibrate_device: size must be a scalar".into(),
            ));
        }
        let v = arr.data()[0];
        if !v.is_finite() || v <= 0.0 {
            return Err(EvalError::Unsupported(format!(
                "calibrate_device: size must be positive, got {v}"
            )));
        }
        v as usize
    } else {
        DEFAULT_BENCH_SIZE
    };
    let gflops = run_matmul_benchmark(env, size)?;
    env.set_string("mlpl_device_throughput_gflops".into(), format!("{gflops}"));
    Ok(DenseArray::from_scalar(gflops))
}

/// Benchmark `BENCH_ITERS + 1` square-matmuls; discard
/// iter 0 as warmup; compute observed GFLOPS from the
/// remaining wall-clock. Uses iota-based fixed inputs
/// to avoid the PRNG dependency and keep the result
/// dominated by arithmetic not randn.
fn run_matmul_benchmark(env: &Environment, size: usize) -> Result<f64, EvalError> {
    let n = size * size;
    let a_data: Vec<f64> = (0..n).map(|i| ((i % 97) as f64) * 0.01).collect();
    let b_data: Vec<f64> = (0..n).map(|i| ((i % 89) as f64) * 0.01).collect();
    let a = DenseArray::new(Shape::new(vec![size, size]), a_data)?;
    let b = DenseArray::new(Shape::new(vec![size, size]), b_data)?;
    let _ = crate::device::dispatched_call(env, "matmul", vec![a.clone(), b.clone()])?;
    let t0 = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = crate::device::dispatched_call(env, "matmul", vec![a.clone(), b.clone()])?;
    }
    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
    let flops = 2.0 * (size as f64).powi(3) * (BENCH_ITERS as f64);
    Ok(flops / elapsed / 1e9)
}

/// `estimate_hypothetical(name, steps, batch, seq [,
/// dtype_bytes, lora_rank]) -> [5]`.
pub(crate) fn eval_estimate_hypothetical(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    if !(4..=6).contains(&args.len()) {
        return Err(EvalError::BadArity {
            func: "estimate_hypothetical".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let Value::Str(name) = crate::eval::eval_expr(&args[0], env, &mut None)? else {
        return Err(EvalError::Unsupported(
            "estimate_hypothetical: first argument must be a model-name string".into(),
        ));
    };
    let steps = pos_scalar(&args[1], env, "steps")?;
    let batch = pos_scalar(&args[2], env, "batch_size")?;
    let seq = pos_scalar(&args[3], env, "seq_len")?;
    let dtype_bytes = if args.len() >= 5 {
        pos_scalar(&args[4], env, "dtype_bytes")?
    } else {
        8.0
    };
    let lora_rank = if args.len() == 6 {
        pos_scalar(&args[5], env, "lora_rank")?
    } else {
        0.0
    };
    let spec = lookup_hypothetical(&name)?;
    let gflops = env
        .get_string("mlpl_device_throughput_gflops")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_GFLOPS);
    Ok(DenseArray::new(
        Shape::new(vec![5]),
        compute_hyp_estimate(&spec, steps, batch, seq, dtype_bytes, lora_rank, gflops),
    )?)
}

fn lookup_hypothetical(name: &str) -> Result<HypSpec, EvalError> {
    let spec = match name {
        "smollm-135m" => HypSpec {
            vocab: 49152.0,
            d_model: 576.0,
            layers: 30.0,
            intermediate: 1536.0,
        },
        "smollm-360m" => HypSpec {
            vocab: 49152.0,
            d_model: 960.0,
            layers: 32.0,
            intermediate: 2560.0,
        },
        "smollm-1.7b" => HypSpec {
            vocab: 49152.0,
            d_model: 2048.0,
            layers: 24.0,
            intermediate: 8192.0,
        },
        "llama-3.2-1b" => HypSpec {
            vocab: 128256.0,
            d_model: 2048.0,
            layers: 16.0,
            intermediate: 8192.0,
        },
        "qwen-2.5-0.5b" => HypSpec {
            vocab: 151936.0,
            d_model: 896.0,
            layers: 24.0,
            intermediate: 4864.0,
        },
        other => {
            return Err(EvalError::Unsupported(format!(
                "estimate_hypothetical: unknown model name '{other}' (try smollm-135m / smollm-360m / smollm-1.7b / llama-3.2-1b / qwen-2.5-0.5b)"
            )));
        }
    };
    Ok(spec)
}

/// Compute `[params, vram, disk, flops, wall]` from
/// the hypothetical spec. Forward pass: embedding
/// gather + `layers` copies of (4 d_model x d_model
/// projections + QK^T + AV + 2 FFN matmuls) + output
/// head. LoRA rank > 0 reclassifies grad + Adam
/// memory onto a reduced trainable set. Activation
/// term uses the same 4x safety factor as
/// `estimate_train`.
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

/// `feasible(est_result, budget) -> 0/1` scalar.
pub(crate) fn eval_feasible(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "feasible".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let est = crate::eval::eval_expr(&args[0], env, &mut None)?.into_array()?;
    let budget = crate::eval::eval_expr(&args[1], env, &mut None)?.into_array()?;
    if est.shape().dims() != [5] {
        return Err(EvalError::Unsupported(format!(
            "feasible: estimate must be rank-1 [5], got {:?}",
            est.shape().dims()
        )));
    }
    if budget.shape().dims() != [3] {
        return Err(EvalError::Unsupported(format!(
            "feasible: budget must be rank-1 [3] [vram, disk, wall], got {:?}",
            budget.shape().dims()
        )));
    }
    let e = est.data();
    let b = budget.data();
    // est layout: [params, vram, disk, flops, wall]
    // budget layout: [vram, disk, wall]; 0 = skip.
    let passes = (b[0] == 0.0 || e[1] <= b[0])
        && (b[1] == 0.0 || e[2] <= b[1])
        && (b[2] == 0.0 || e[4] <= b[2]);
    Ok(DenseArray::from_scalar(if passes { 1.0 } else { 0.0 }))
}

fn pos_scalar(arg: &Expr, env: &mut Environment, name: &str) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(arg, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "estimate_hypothetical: {name} must be a scalar"
        )));
    }
    let v = arr.data()[0];
    if !v.is_finite() || v <= 0.0 {
        return Err(EvalError::Unsupported(format!(
            "estimate_hypothetical: {name} must be positive, got {v}"
        )));
    }
    Ok(v)
}
