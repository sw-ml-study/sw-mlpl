use mlpl_array::{DenseArray, Shape};
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

pub(crate) fn constructor(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if name == "fill" {
        return fill_constructor(name, args);
    }
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

fn fill_constructor(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
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
}

pub(crate) fn array_util(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if name == "concat" {
        concat_1d(name, args)
    } else {
        last_row(name, args)
    }
}

fn concat_1d(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    for (i, a) in args.iter().enumerate() {
        if a.rank() > 1 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("argument {i} must be rank 0 or 1, got rank {}", a.rank()),
            });
        }
    }
    let mut data = Vec::with_capacity(args[0].data().len() + args[1].data().len());
    data.extend_from_slice(args[0].data());
    data.extend_from_slice(args[1].data());
    Ok(DenseArray::from_vec(data))
}

fn last_row(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    if args[0].rank() != 2 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("expected rank-2 matrix, got rank {}", args[0].rank()),
        });
    }
    let cols = args[0].shape().dims()[1];
    let data = args[0].data();
    Ok(DenseArray::from_vec(data[data.len() - cols..].to_vec()))
}
