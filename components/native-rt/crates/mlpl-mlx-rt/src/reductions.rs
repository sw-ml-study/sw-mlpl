//! MLX-backed reductions, normalisation, and loss primitives
//! (Saga 14 step 003).
//!
//! Signatures mirror `mlpl-rt::reductions` so compiled MLPL code
//! can swap runtimes without source changes:
//!
//! - `reduce_mul` / `mean` / `argmax` take `Option<usize>` axis
//!   (None = flat reduction over the whole array).
//! - `softmax` / `log_softmax` take `usize` axis.
//! - `cross_entropy(logits, targets)` is a fused log-softmax + NLL
//!   that returns an unlabeled scalar mean.
//!
//! Validation (axis bounds, target integrality, target range) lives
//! on the Rust side so MLX sees only legal inputs. Numerical work
//! routes through MLX's own kernels (`prod`, `mean`, `argmax`,
//! `softmax_axis`, `log_softmax`, `logsumexp_axis`); the fp32
//! GPU/Accelerate path means parity tests use a documented
//! tolerance rather than bit-for-bit equality. Label propagation
//! follows Saga 11.5: reductions drop the reduced axis label,
//! softmax/log_softmax preserve labels, cross_entropy returns
//! unlabeled.

use mlpl_array::{ArrayError, DenseArray, Shape};

use crate::common::{dense_to_mlx, finalize, mlx_to_dense_data};

/// Product reduction over `axis` (None = flat).
pub fn reduce_mul(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    let mlx = dense_to_mlx(a.data(), a.shape().dims());
    if let Some(ax) = axis {
        let dims = a.shape().dims();
        if ax >= dims.len() {
            return Err(ArrayError::IndexOutOfBounds {
                axis: ax,
                index: ax,
                size: dims.len(),
            });
        }
        let reduced = mlx
            .prod_axis(ax as i32, false)
            .expect("mlx prod_axis on validated axis");
        let (out_dims, labels) = drop_axis(a, ax);
        return finalize(Shape::new(out_dims), mlx_to_dense_data(reduced), labels);
    }
    let reduced = mlx.prod(false).expect("mlx prod on validated array");
    finalize(Shape::new(vec![]), mlx_to_dense_data(reduced), None)
}

/// Mean reduction over `axis` (None = flat).
pub fn mean(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    let mlx = dense_to_mlx(a.data(), a.shape().dims());
    if let Some(ax) = axis {
        let dims = a.shape().dims();
        if ax >= dims.len() {
            return Err(ArrayError::IndexOutOfBounds {
                axis: ax,
                index: ax,
                size: dims.len(),
            });
        }
        let reduced = mlx
            .mean_axis(ax as i32, false)
            .expect("mlx mean_axis on validated axis");
        let (out_dims, labels) = drop_axis(a, ax);
        return finalize(Shape::new(out_dims), mlx_to_dense_data(reduced), labels);
    }
    let reduced = mlx.mean(false).expect("mlx mean on validated array");
    finalize(Shape::new(vec![]), mlx_to_dense_data(reduced), None)
}

/// Index of the maximum value over `axis` (None = flat). The
/// MLX-side argmax returns an unsigned integer array; we cast it
/// to f32 so the f64 round trip in `mlx_to_dense_data` agrees with
/// the CPU path's `f64` index representation.
pub fn argmax(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    if a.elem_count() == 0 && axis.is_none() {
        return Err(ArrayError::EmptyArray);
    }
    let mlx = dense_to_mlx(a.data(), a.shape().dims());
    let raw = if let Some(ax) = axis {
        let dims = a.shape().dims();
        if ax >= dims.len() {
            return Err(ArrayError::IndexOutOfBounds {
                axis: ax,
                index: ax,
                size: dims.len(),
            });
        }
        mlx_rs::ops::indexing::argmax_axis(&mlx, ax as i32, false)
            .expect("mlx argmax_axis on validated axis")
    } else {
        mlx_rs::ops::indexing::argmax(&mlx, false).expect("mlx argmax on non-empty array")
    };
    let as_f32 = raw
        .as_type::<f32>()
        .expect("argmax result casts to f32 for f64 round trip");
    let (out_dims, labels) = match axis {
        Some(ax) => drop_axis(a, ax),
        None => (vec![], None),
    };
    finalize(Shape::new(out_dims), mlx_to_dense_data(as_f32), labels)
}

/// `softmax(a, axis)` via MLX's stable kernel. Output shape and
/// labels match the input.
pub fn softmax(a: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
    let dims = a.shape().dims().to_vec();
    if axis >= dims.len() {
        return Err(ArrayError::IndexOutOfBounds {
            axis,
            index: axis,
            size: dims.len(),
        });
    }
    let mlx = dense_to_mlx(a.data(), &dims);
    let result = mlx_rs::ops::softmax_axis(&mlx, axis as i32, false)
        .expect("mlx softmax_axis on validated input");
    finalize(
        Shape::new(dims),
        mlx_to_dense_data(result),
        a.labels().map(<[Option<String>]>::to_vec),
    )
}

/// `log_softmax(a, axis)` via MLX's `nn::log_softmax`, which is
/// implemented as `x - logsumexp_axis(x, axis, true)` -- the same
/// max-subtraction LSE form as the CPU path.
pub fn log_softmax(a: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
    let dims = a.shape().dims().to_vec();
    if axis >= dims.len() {
        return Err(ArrayError::IndexOutOfBounds {
            axis,
            index: axis,
            size: dims.len(),
        });
    }
    let mlx = dense_to_mlx(a.data(), &dims);
    let result =
        mlx_rs::nn::log_softmax(&mlx, axis as i32).expect("mlx log_softmax on validated input");
    finalize(
        Shape::new(dims),
        mlx_to_dense_data(result),
        a.labels().map(<[Option<String>]>::to_vec),
    )
}

/// `cross_entropy(logits, targets)`: fused log-softmax + NLL with
/// scalar mean output. Validation matches the CPU path exactly;
/// the fp32 round-trip happens twice (once for the LSE pass on
/// MLX, once when reading the per-row LSE back into Rust for the
/// gather-and-mean) so the parity tolerance accumulates a small
/// constant factor over a single MLX op.
pub fn cross_entropy(logits: &DenseArray, targets: &DenseArray) -> Result<DenseArray, ArrayError> {
    let dims = logits.shape().dims();
    let (n, v) = match dims.len() {
        2 => (dims[0], dims[1]),
        3 => (dims[0] * dims[1], dims[2]),
        r => {
            return Err(ArrayError::RankMismatch {
                expected: 2,
                got: r,
            });
        }
    };
    if targets.elem_count() != n {
        return Err(ArrayError::ShapeMismatch {
            source: n,
            target: targets.elem_count(),
        });
    }
    // Validate target integrality and range up front.
    let mut idx = Vec::with_capacity(n);
    for (i, &t) in targets.data().iter().enumerate() {
        if t < 0.0 || t.fract() != 0.0 {
            return Err(ArrayError::IndexOutOfBounds {
                axis: 0,
                index: i,
                size: v,
            });
        }
        let ti = t as usize;
        if ti >= v {
            return Err(ArrayError::IndexOutOfBounds {
                axis: 0,
                index: ti,
                size: v,
            });
        }
        idx.push(ti);
    }
    // logits as [N, V] for a single MLX kernel call.
    let mlx = dense_to_mlx(logits.data(), &[n, v]);
    let lse =
        mlx_rs::ops::logsumexp_axis(&mlx, 1, false).expect("mlx logsumexp_axis on [N, V] logits");
    let lse_data = mlx_to_dense_data(lse); // length N, fp32 round-tripped.
    let data = logits.data();
    let mut total = 0.0;
    for (i, &ti) in idx.iter().enumerate() {
        total += lse_data[i] - data[i * v + ti];
    }
    Ok(DenseArray::from_scalar(total / n as f64))
}

/// Drop axis `ax` from `a`'s dims and labels. Helper used by every
/// axis-aware reduction so the bookkeeping lives in one place.
fn drop_axis(a: &DenseArray, ax: usize) -> (Vec<usize>, Option<Vec<Option<String>>>) {
    let mut dims: Vec<usize> = a.shape().dims().to_vec();
    dims.remove(ax);
    let labels = a.labels().map(|lbls| {
        let mut out = lbls.to_vec();
        out.remove(ax);
        out
    });
    (dims, labels)
}
