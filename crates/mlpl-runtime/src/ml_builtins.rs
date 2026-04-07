//! Higher-level ML built-ins: `softmax` and `one_hot`.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;

/// Dispatch ML built-ins. Returns None if not matched.
pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "softmax" => Some(builtin_softmax(name, args)),
        "one_hot" => Some(builtin_one_hot(name, args)),
        _ => None,
    }
}

fn builtin_softmax(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
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
            reason: format!("axis must be scalar, got rank {}", args[1].rank()),
        });
    }
    let a = &args[0];
    let axis = args[1].data()[0] as usize;
    let dims = a.shape().dims().to_vec();
    if axis >= dims.len() {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("axis {axis} out of range for rank {}", dims.len()),
        });
    }

    // Strides for the input shape.
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    let axis_size = dims[axis];
    let axis_stride = strides[axis];

    // Group index: flat index with the `axis` coordinate removed.
    let mut group_dims = dims.clone();
    group_dims.remove(axis);
    let group_count: usize = group_dims.iter().product::<usize>().max(1);

    // First pass: per-group max for numerical stability.
    let mut maxv = vec![f64::NEG_INFINITY; group_count];
    for flat in 0..a.elem_count() {
        let g = if axis_stride > 1 {
            let outer = flat / (axis_size * axis_stride);
            let inner = flat % axis_stride;
            outer * axis_stride + inner
        } else {
            flat / axis_size
        };
        let v = a.data()[flat];
        if v > maxv[g] {
            maxv[g] = v;
        }
    }

    // Second pass: exponentiate shifted logits and accumulate per-group sums.
    let mut out = vec![0.0f64; a.elem_count()];
    let mut sums = vec![0.0f64; group_count];
    for (flat, slot) in out.iter_mut().enumerate() {
        let g = if axis_stride > 1 {
            let outer = flat / (axis_size * axis_stride);
            let inner = flat % axis_stride;
            outer * axis_stride + inner
        } else {
            flat / axis_size
        };
        let e = (a.data()[flat] - maxv[g]).exp();
        *slot = e;
        sums[g] += e;
    }

    // Third pass: normalize.
    for (flat, slot) in out.iter_mut().enumerate() {
        let g = if axis_stride > 1 {
            let outer = flat / (axis_size * axis_stride);
            let inner = flat % axis_stride;
            outer * axis_stride + inner
        } else {
            flat / axis_size
        };
        *slot /= sums[g];
    }

    Ok(DenseArray::new(Shape::new(dims), out)?)
}

fn builtin_one_hot(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
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
    let labels = args[0].data();
    let k = args[1].data()[0] as usize;
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
