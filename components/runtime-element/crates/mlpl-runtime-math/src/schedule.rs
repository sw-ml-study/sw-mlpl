//! Learning-rate SCHEDULES: `cosine_schedule(step, total, lr_min, lr_max)`
//! and `linear_warmup(step, warmup, lr)`. Each takes scalar arguments and
//! returns a scalar learning rate.

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub(crate) fn schedule(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let expected = if name == "cosine_schedule" { 4 } else { 3 };
    validate_schedule_args(name, &args, expected)?;
    let v = compute_schedule(name, &args);
    Ok(DenseArray::from_scalar(v))
}

fn validate_schedule_args(
    name: &str,
    args: &[DenseArray],
    expected: usize,
) -> Result<(), RuntimeError> {
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
    Ok(())
}

fn compute_schedule(name: &str, args: &[DenseArray]) -> f64 {
    if name == "cosine_schedule" {
        let step = args[0].data()[0];
        let total = args[1].data()[0];
        let lr_min = args[2].data()[0];
        let lr_max = args[3].data()[0];
        if total <= 0.0 {
            return lr_max;
        }
        let t = step.clamp(0.0, total) / total;
        lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
    } else {
        let step = args[0].data()[0];
        let warmup = args[1].data()[0];
        let lr = args[2].data()[0];
        if warmup <= 0.0 {
            return lr;
        }
        lr * (step / warmup).clamp(0.0, 1.0)
    }
}
