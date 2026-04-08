//! Math and constructor built-in functions.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;

/// Dispatch math and constructor built-ins. Returns None if not matched.
pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "exp" => Some(unary(name, args, f64::exp)),
        "log" => Some(unary(name, args, f64::ln)),
        "sqrt" => Some(unary(name, args, f64::sqrt)),
        "abs" => Some(unary(name, args, f64::abs)),
        "sigmoid" => Some(unary(name, args, |x| 1.0 / (1.0 + (-x).exp()))),
        "tanh_fn" => Some(unary(name, args, f64::tanh)),
        "pow" => Some(binary_pow(name, args)),
        "gt" => Some(binary_cmp(name, args, |a, b| if a > b { 1.0 } else { 0.0 })),
        "lt" => Some(binary_cmp(name, args, |a, b| if a < b { 1.0 } else { 0.0 })),
        "eq" => Some(binary_cmp(name, args, |a, b| {
            if (a - b).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        })),
        "mean" => Some(builtin_mean(name, args)),
        "zeros" | "ones" | "fill" => Some(constructor(name, args)),
        "cosine_schedule" | "linear_warmup" => Some(schedule(name, args)),
        _ => None,
    }
}

fn unary(name: &str, args: Vec<DenseArray>, f: fn(f64) -> f64) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    Ok(args[0].map(f))
}

fn binary_pow(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    Ok(args[0].apply_binop(&args[1], f64::powf)?)
}

fn binary_cmp(
    name: &str,
    args: Vec<DenseArray>,
    f: fn(f64, f64) -> f64,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    Ok(args[0].apply_binop(&args[1], f)?)
}

fn builtin_mean(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let data = args[0].data();
    let sum: f64 = data.iter().sum();
    let count = data.len().max(1) as f64;
    Ok(DenseArray::from_scalar(sum / count))
}

/// Pure scalar learning-rate schedule built-ins.
///
/// - `cosine_schedule(step, total, lr_min, lr_max)`:
///   `lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * step/total))`,
///   clamped to `[lr_min, lr_max]` for `step` outside `[0, total]`.
/// - `linear_warmup(step, warmup, lr)`:
///   `lr * min(1, step/warmup)`, clamped to `[0, lr]`.
fn schedule(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let expected = if name == "cosine_schedule" { 4 } else { 3 };
    if args.len() != expected {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected,
            got: args.len(),
        });
    }
    for (i, a) in args.iter().enumerate() {
        if a.rank() != 0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("argument {i} must be a scalar, got rank {}", a.rank()),
            });
        }
    }
    let v = if name == "cosine_schedule" {
        let step = args[0].data()[0];
        let total = args[1].data()[0];
        let lr_min = args[2].data()[0];
        let lr_max = args[3].data()[0];
        if total <= 0.0 {
            lr_max
        } else {
            let t = step.clamp(0.0, total) / total;
            lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
        }
    } else {
        let step = args[0].data()[0];
        let warmup = args[1].data()[0];
        let lr = args[2].data()[0];
        if warmup <= 0.0 {
            lr
        } else {
            lr * (step / warmup).clamp(0.0, 1.0)
        }
    };
    Ok(DenseArray::from_scalar(v))
}

fn constructor(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if name == "fill" {
        if args.len() != 2 {
            return Err(RuntimeError::ArityMismatch {
                func: name.into(),
                expected: 2,
                got: args.len(),
            });
        }
        if args[1].rank() != 0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: "value must be scalar".into(),
            });
        }
        let dims: Vec<usize> = args[0].data().iter().map(|&d| d as usize).collect();
        let count = dims.iter().product();
        Ok(DenseArray::new(
            Shape::new(dims),
            vec![args[1].data()[0]; count],
        )?)
    } else {
        if args.len() != 1 {
            return Err(RuntimeError::ArityMismatch {
                func: name.into(),
                expected: 1,
                got: args.len(),
            });
        }
        let dims: Vec<usize> = args[0].data().iter().map(|&d| d as usize).collect();
        let val = if name == "zeros" { 0.0 } else { 1.0 };
        let count = dims.iter().product();
        Ok(DenseArray::new(Shape::new(dims), vec![val; count])?)
    }
}
