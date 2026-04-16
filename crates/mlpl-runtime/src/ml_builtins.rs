//! Higher-level ML built-ins: `softmax`, `one_hot`, `sinusoidal_encoding`.

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
        "sinusoidal_encoding" => Some(builtin_sinusoidal_encoding(name, args)),
        "cross_entropy" => Some(builtin_cross_entropy(name, args)),
        _ => None,
    }
}

/// `cross_entropy(logits, targets)` fused log-softmax + NLL, scalar mean.
///
/// Logits are `[N, V]` or `[B, T, V]` float; targets are `[N]` or
/// `[B, T]` integer-valued (represented as f64 with whole-number
/// values, the convention MLPL uses for all integer-typed builtins).
/// Errors surface as `InvalidArgument` with the reason string; the
/// `mlpl-eval` layer maps them to `EvalError::ShapeMismatch` where
/// appropriate.
fn builtin_cross_entropy(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    let (n, v, idx) = cross_entropy_indices(name, &args[0], &args[1])?;
    let data = args[0].data();
    let mut total = 0.0;
    for (i, &t) in idx.iter().enumerate() {
        let row = &data[i * v..(i + 1) * v];
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse: f64 = m + row.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        total += lse - row[t];
    }
    Ok(DenseArray::from_scalar(total / n as f64))
}

/// Validate logits / targets shapes and convert targets into a flat
/// `Vec<usize>`. Returns `(N, V, indices)` on success.
fn cross_entropy_indices(
    name: &str,
    logits: &DenseArray,
    targets: &DenseArray,
) -> Result<(usize, usize, Vec<usize>), RuntimeError> {
    let dims = logits.shape().dims();
    let (n, v) = match dims.len() {
        2 => (dims[0], dims[1]),
        3 => (dims[0] * dims[1], dims[2]),
        r => {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("logits must be rank 2 or 3, got rank {r}"),
            });
        }
    };
    if targets.elem_count() != n {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!(
                "targets must have {n} elements to match logits rows, got {}",
                targets.elem_count()
            ),
        });
    }
    let mut idx = Vec::with_capacity(n);
    for (i, &t) in targets.data().iter().enumerate() {
        if t < 0.0 || t.fract() != 0.0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("target[{i}] must be a non-negative integer, got {t}"),
            });
        }
        let ti = t as usize;
        if ti >= v {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("target[{i}] = {ti} out of range for V = {v}"),
            });
        }
        idx.push(ti);
    }
    Ok((n, v, idx))
}

/// Standard transformer sinusoidal positional encoding table.
///
/// `sinusoidal_encoding(seq_len, d_model)` returns a deterministic
/// `[seq_len, d_model]` table labeled `[time, dim]`:
///
/// - even cols `2i`:   `sin(pos / 10000^(2i / d_model))`
/// - odd cols `2i+1`:  `cos(pos / 10000^(2i / d_model))`
///
/// Pure -- no params, no seed. The `[time, dim]` labels propagate
/// through element-wise add when paired with a token embedding
/// output of matching shape.
fn builtin_sinusoidal_encoding(
    name: &str,
    args: Vec<DenseArray>,
) -> Result<DenseArray, RuntimeError> {
    let (t, d) = parse_sinusoidal_args(name, &args)?;
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

fn parse_sinusoidal_args(name: &str, args: &[DenseArray]) -> Result<(usize, usize), RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
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
    let seq_len = args[0].data()[0];
    let d_model = args[1].data()[0];
    if seq_len < 0.0 || seq_len.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("seq_len must be a non-negative integer, got {seq_len}"),
        });
    }
    if d_model <= 0.0 || d_model.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("d_model must be a positive integer, got {d_model}"),
        });
    }
    Ok((seq_len as usize, d_model as usize))
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
