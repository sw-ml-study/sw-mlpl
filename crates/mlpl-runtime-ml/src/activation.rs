use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

struct SoftmaxParams {
    dims: Vec<usize>,
    axis_size: usize,
    axis_stride: usize,
}

/// `softmax(input, axis)` -- numerically stable, axis-aware softmax.
pub(crate) fn softmax(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let p = validate_softmax_args(name, &args)?;
    let out = compute_softmax(args[0].data(), &p);
    Ok(DenseArray::new(Shape::new(p.dims), out)?)
}

fn validate_softmax_args(name: &str, args: &[DenseArray]) -> Result<SoftmaxParams, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("axis must be scalar, got rank {}", args[1].rank()),
        });
    }
    let axis = args[1].data()[0] as usize;
    let dims = args[0].shape().dims().to_vec();
    if axis >= dims.len() {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("axis {axis} out of range for rank {}", dims.len()),
        });
    }
    let axis_size = dims[axis];
    let axis_stride: usize = dims[axis + 1..].iter().product();
    Ok(SoftmaxParams {
        dims,
        axis_size,
        axis_stride,
    })
}

fn compute_softmax(data: &[f64], p: &SoftmaxParams) -> Vec<f64> {
    let group_count = (data.len() / p.axis_size).max(1);
    let mut maxv = vec![f64::NEG_INFINITY; group_count];
    for (flat, &v) in data.iter().enumerate() {
        let g = group_index(flat, p.axis_size, p.axis_stride);
        if v > maxv[g] {
            maxv[g] = v;
        }
    }
    let mut out = vec![0.0f64; data.len()];
    let mut sums = vec![0.0f64; group_count];
    for (flat, slot) in out.iter_mut().enumerate() {
        let g = group_index(flat, p.axis_size, p.axis_stride);
        let e = (data[flat] - maxv[g]).exp();
        *slot = e;
        sums[g] += e;
    }
    for (flat, slot) in out.iter_mut().enumerate() {
        let g = group_index(flat, p.axis_size, p.axis_stride);
        *slot /= sums[g];
    }
    out
}

/// Map a flat element index to its softmax group index --
/// the same flat index with the `axis` coordinate stripped out.
fn group_index(flat: usize, axis_size: usize, axis_stride: usize) -> usize {
    if axis_stride > 1 {
        let outer = flat / (axis_size * axis_stride);
        let inner = flat % axis_stride;
        outer * axis_stride + inner
    } else {
        flat / axis_size
    }
}
