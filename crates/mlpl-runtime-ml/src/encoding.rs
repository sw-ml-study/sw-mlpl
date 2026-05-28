use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

/// Standard transformer sinusoidal positional encoding table.
/// `sinusoidal_encoding(seq_len, d_model)` returns a `[seq_len, d_model]`
/// table labeled `[time, dim]` with:
/// - even cols `2i`:   `sin(pos / 10000^(2i / d_model))`
/// - odd cols `2i+1`:  `cos(pos / 10000^(2i / d_model))`
pub(crate) fn sinusoidal(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (t, d) = parse_sinusoidal_dims(name, &args)?;
    let mut data = vec![0.0_f64; t * d];
    for pos in 0..t {
        for col in 0..d {
            let exponent = (2 * (col / 2)) as f64 / d as f64;
            let theta = pos as f64 / 10000.0_f64.powf(exponent);
            data[pos * d + col] = if col % 2 == 0 {
                theta.sin()
            } else {
                theta.cos()
            };
        }
    }
    let arr = DenseArray::new(Shape::new(vec![t, d]), data)?;
    Ok(arr.with_labels(vec![Some("time".into()), Some("dim".into())])?)
}

fn parse_sinusoidal_dims(name: &str, args: &[DenseArray]) -> Result<(usize, usize), RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    for (i, a) in args.iter().enumerate() {
        if a.rank() != 0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("argument {i} must be a scalar, got rank {}", a.rank()),
            });
        }
    }
    let seq_len = nonneg_int(name, args[0].data()[0], "seq_len")?;
    let d_model = pos_int(name, args[1].data()[0], "d_model")?;
    Ok((seq_len, d_model))
}

fn nonneg_int(name: &str, v: f64, what: &str) -> Result<usize, RuntimeError> {
    if v < 0.0 || v.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("{what} must be a non-negative integer, got {v}"),
        });
    }
    Ok(v as usize)
}

fn pos_int(name: &str, v: f64, what: &str) -> Result<usize, RuntimeError> {
    if v <= 0.0 || v.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("{what} must be a positive integer, got {v}"),
        });
    }
    Ok(v as usize)
}

/// `one_hot(labels, k)` -- `labels: [N]` integer-valued vector,
/// `k: scalar`. Returns `[N, k]` with a 1.0 at column `labels[i]`
/// in each row and zeros elsewhere.
pub(crate) fn one_hot(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (labels, k) = parse_one_hot_args(name, &args)?;
    let n = labels.len();
    let mut data = vec![0.0f64; n * k];
    for (i, &lab) in labels.iter().enumerate() {
        let j = lab as usize;
        if j >= k {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("label {j} out of range for k = {k}"),
            });
        }
        data[i * k + j] = 1.0;
    }
    Ok(DenseArray::new(Shape::new(vec![n, k]), data)?)
}

fn parse_one_hot_args<'a>(
    name: &str,
    args: &'a [DenseArray],
) -> Result<(&'a [f64], usize), RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    if args[0].rank() != 1 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("labels must be a vector, got rank {}", args[0].rank()),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "k must be a scalar".into(),
        });
    }
    Ok((args[0].data(), args[1].data()[0] as usize))
}
