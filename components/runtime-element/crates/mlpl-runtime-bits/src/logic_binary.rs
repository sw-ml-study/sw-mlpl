//! Element-wise bitwise BINARY ops (band / bor / bxor) with scalar
//! broadcast (docs/bit-ops-design.md). The `op` is a plain `u64` combiner;
//! `binary` checks arity and delegates the shape handling to `zip_bits`.

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;
use crate::bit_domain::as_uint;

/// Element-wise bitwise binary op with scalar broadcast.
fn zip_bits(
    name: &str,
    a: &DenseArray,
    b: &DenseArray,
    op: fn(u64, u64) -> u64,
) -> Result<DenseArray, RuntimeError> {
    let (data, shape) = if a.shape() == b.shape() {
        (broadcast_none(name, a, b, op)?, a.shape().clone())
    } else if a.rank() == 0 {
        (
            broadcast_scalar(name, a.data()[0], b, op, true)?,
            b.shape().clone(),
        )
    } else if b.rank() == 0 {
        (
            broadcast_scalar(name, b.data()[0], a, op, false)?,
            a.shape().clone(),
        )
    } else {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!(
                "shape mismatch: {:?} vs {:?}",
                a.shape().dims(),
                b.shape().dims()
            ),
        });
    };
    Ok(DenseArray::new(shape, data)?)
}

fn broadcast_none(
    name: &str,
    a: &DenseArray,
    b: &DenseArray,
    op: fn(u64, u64) -> u64,
) -> Result<Vec<f64>, RuntimeError> {
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| Ok(op(as_uint(name, *x)?, as_uint(name, *y)?) as f64))
        .collect()
}

fn broadcast_scalar(
    name: &str,
    s: f64,
    arr: &DenseArray,
    op: fn(u64, u64) -> u64,
    scalar_left: bool,
) -> Result<Vec<f64>, RuntimeError> {
    let su = as_uint(name, s)?;
    arr.data()
        .iter()
        .map(|x| {
            let xu = as_uint(name, *x)?;
            let (l, r) = if scalar_left { (su, xu) } else { (xu, su) };
            Ok(op(l, r) as f64)
        })
        .collect()
}

/// Arity-checked entry point for the binary logic builtins.
pub(crate) fn binary(
    name: &str,
    args: Vec<DenseArray>,
    op: fn(u64, u64) -> u64,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    zip_bits(name, &args[0], &args[1], op)
}
