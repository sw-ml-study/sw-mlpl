//! `calibrate_device([size])` -- benchmark matmul on the
//! active device and cache observed GFLOPS into
//! `mlpl_device_throughput_gflops`. The first sub-crate fn
//! that uses the `HasDispatch` behavior trait to drive
//! device-aware ops.

use std::time::Instant;

use mlpl_array::{DenseArray, Shape};
use mlpl_env_traits::{HasDispatch, HasStrings};
use mlpl_parser::Expr;

use crate::error::FeasibilityError;

const DEFAULT_BENCH_SIZE: usize = 1024;
const BENCH_ITERS: u32 = 10;

pub fn calibrate_device_inner<E, S, Err>(
    args: &[Expr],
    env: &mut E,
    scalar: S,
) -> Result<DenseArray, Err>
where
    E: HasDispatch + HasStrings,
    S: FnOnce(&Expr, &mut E) -> Result<f64, Err>,
    Err: From<FeasibilityError>,
{
    if args.len() > 1 {
        return Err(FeasibilityError::BadArity {
            func: "calibrate_device".into(),
            expected: 0,
            got: args.len(),
        }
        .into());
    }
    let size = parse_size(args, env, scalar)?;
    let gflops = run_matmul_benchmark(env, size)?;
    env.set_string("mlpl_device_throughput_gflops".into(), format!("{gflops}"));
    Ok(DenseArray::from_scalar(gflops))
}

fn parse_size<E, S, Err>(args: &[Expr], env: &mut E, scalar: S) -> Result<usize, Err>
where
    S: FnOnce(&Expr, &mut E) -> Result<f64, Err>,
    Err: From<FeasibilityError>,
{
    let Some(arg) = args.first() else {
        return Ok(DEFAULT_BENCH_SIZE);
    };
    let v = scalar(arg, env)?;
    if !v.is_finite() || v <= 0.0 {
        return Err(FeasibilityError::NotPositive {
            func: "calibrate_device".into(),
            name: "size".into(),
            value: v,
        }
        .into());
    }
    Ok(v as usize)
}

fn run_matmul_benchmark<E: HasDispatch>(env: &E, size: usize) -> Result<f64, FeasibilityError> {
    let n = size * size;
    let a_data: Vec<f64> = (0..n).map(|i| ((i % 97) as f64) * 0.01).collect();
    let b_data: Vec<f64> = (0..n).map(|i| ((i % 89) as f64) * 0.01).collect();
    let a = DenseArray::new(Shape::new(vec![size, size]), a_data)?;
    let b = DenseArray::new(Shape::new(vec![size, size]), b_data)?;
    let _ = env.dispatch("matmul", vec![a.clone(), b.clone()])?;
    let t0 = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = env.dispatch("matmul", vec![a.clone(), b.clone()])?;
    }
    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
    let flops = 2.0 * (size as f64).powi(3) * f64::from(BENCH_ITERS);
    Ok(flops / elapsed / 1e9)
}
