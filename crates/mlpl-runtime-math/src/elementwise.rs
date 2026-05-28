use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub(crate) fn zero_arg(
    name: &str,
    args: Vec<DenseArray>,
    val: f64,
) -> Result<DenseArray, RuntimeError> {
    if !args.is_empty() {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 0,
            got: args.len(),
        });
    }
    Ok(DenseArray::from_scalar(val))
}

pub(crate) fn unary(
    name: &str,
    args: Vec<DenseArray>,
    f: fn(f64) -> f64,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    Ok(args[0].map(f))
}

pub(crate) fn binary_pow(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    Ok(args[0].apply_binop(&args[1], f64::powf)?)
}

pub(crate) fn binary_cmp(
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
