//! MLX-backed primitive implementations.
//!
//! Only compiled when the `mlx` feature is on AND the host is
//! Apple Silicon macOS (`mlx-rs` links against MLX's C++/Metal
//! runtime, which is Apple-only). See `crate::lib` for the
//! rationale behind the three-way cfg gate.
//!
//! Phase 1 step 001 ports exactly one primitive: `matmul`.
//! Shape validation, rank checking, and label propagation mirror
//! `mlpl_array::DenseArray::matmul` so the CPU and MLX paths are
//! observationally identical on success *and* failure -- the only
//! difference is that MLX computes the data in fp32 on the Apple
//! GPU / Accelerate path and we cast back to f64 on the way out.

use mlpl_array::{ArrayError, DenseArray, Shape};
use mlx_rs::Array as MlxArray;

/// Matrix multiply backed by MLX.
///
/// Shape rules match the CPU path:
/// - `[m, k] @ [k, n] -> [m, n]`
/// - `[m, k] @ [k]    -> [m]`
///
/// Labels propagate identically: the contraction axis (`a`'s last
/// dim vs `b`'s first dim) must agree when both sides are labeled,
/// and the surviving label list is `[a.labels[0], b.labels[1]]`
/// (matrix-matrix) or `[a.labels[0]]` (matrix-vector). See
/// `mlpl_array::DenseArray::matmul` for the authoritative spec.
///
/// MLX inputs go through the GPU/Accelerate path in fp32, so the
/// parity test in `tests/parity_tests.rs` is tolerance-bounded
/// (f32 round-trip error), not bit-for-bit.
pub fn matmul(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    let (m, k) = match a.shape().dims() {
        [m, k] => (*m, *k),
        _ => {
            return Err(ArrayError::RankMismatch {
                expected: 2,
                got: a.rank(),
            });
        }
    };
    let result_labels = matmul_labels(a, b)?;
    match b.shape().dims() {
        [k2, n] if *k2 == k => {
            let n = *n;
            let data = mlx_matmul_compute(a.data(), b.data(), &[m, k], &[k, n]);
            finalize(Shape::new(vec![m, n]), data, result_labels)
        }
        [k2] if *k2 == k => {
            let data = mlx_matmul_compute(a.data(), b.data(), &[m, k], &[k]);
            finalize(Shape::vector(m), data, result_labels)
        }
        _ => Err(ArrayError::ShapeMismatch {
            source: k,
            target: b.shape().dims().first().copied().unwrap_or(0),
        }),
    }
}

/// Run the actual MLX matmul on two shape-validated f64 buffers.
///
/// Casts f64 -> f32 on the way in (MLX's GPU path is fp32) and
/// f32 -> f64 on the way out. The `.eval()` call forces MLX's
/// lazy graph so `as_slice` sees materialized data.
fn mlx_matmul_compute(a: &[f64], b: &[f64], a_dims: &[usize], b_dims: &[usize]) -> Vec<f64> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let a_shape: Vec<i32> = a_dims.iter().map(|&d| d as i32).collect();
    let b_shape: Vec<i32> = b_dims.iter().map(|&d| d as i32).collect();
    let a_mlx = MlxArray::from_slice(&a_f32, &a_shape);
    let b_mlx = MlxArray::from_slice(&b_f32, &b_shape);
    let c = a_mlx
        .matmul(&b_mlx)
        .expect("mlx matmul on pre-validated shapes should not fail");
    c.eval()
        .expect("mlx eval on pre-validated shapes should not fail");
    let out: &[f32] = c.as_slice();
    out.iter().map(|&x| x as f64).collect()
}

/// Attach labels and wrap the computed data as a `DenseArray`.
fn finalize(
    shape: Shape,
    data: Vec<f64>,
    labels: Option<Vec<Option<String>>>,
) -> Result<DenseArray, ArrayError> {
    let array = DenseArray::new(shape, data)?;
    match labels {
        Some(lbls) => array.with_labels(lbls),
        None => Ok(array),
    }
}

/// Compute the result labels for matmul, mirroring
/// `mlpl_array::ops::matmul_labels` byte-for-byte (Saga 11.5
/// Phase 3 semantics). Kept local rather than re-exported from
/// `mlpl-array` so the CPU crate's internal helper stays internal.
fn matmul_labels(
    a: &DenseArray,
    b: &DenseArray,
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.labels().is_none() && b.labels().is_none() {
        return Ok(None);
    }
    let default_b = vec![None; b.rank()];
    let al: &[Option<String>] = a.labels().unwrap_or(&[None, None][..]);
    let bl: &[Option<String>] = b.labels().unwrap_or(default_b.as_slice());
    if let (Some(sa), Some(sb)) = (&al[1], &bl[0])
        && sa != sb
    {
        return Err(ArrayError::LabelMismatch {
            expected: al.to_vec(),
            actual: bl.to_vec(),
        });
    }
    let mut result = vec![al[0].clone()];
    if b.rank() == 2 {
        result.push(bl[1].clone());
    }
    Ok(Some(result))
}
