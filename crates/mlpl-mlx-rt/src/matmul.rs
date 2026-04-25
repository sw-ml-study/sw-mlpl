//! MLX-backed matrix multiply (Saga 14 step 001).
//!
//! Shape-validated on the Rust side so MLX sees only legal inputs;
//! label propagation mirrors `mlpl_array::DenseArray::matmul`
//! (Saga 11.5 Phase 3). The fp32 GPU/Accelerate path means the
//! parity test in `tests/parity_tests.rs` uses a tolerance bound
//! rather than bit-for-bit equality -- see that module's docstring.

use mlpl_array::{ArrayError, DenseArray, Shape};

use crate::common::{dense_to_mlx, finalize, matmul_labels, mlx_to_dense_data};

/// Matrix multiply backed by MLX.
///
/// Shape rules match the CPU path:
/// - `[m, k] @ [k, n] -> [m, n]`
/// - `[m, k] @ [k]    -> [m]`
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
            let data = compute(a.data(), b.data(), &[m, k], &[k, n]);
            finalize(Shape::new(vec![m, n]), data, result_labels)
        }
        [k2] if *k2 == k => {
            let data = compute(a.data(), b.data(), &[m, k], &[k]);
            finalize(Shape::vector(m), data, result_labels)
        }
        _ => Err(ArrayError::ShapeMismatch {
            source: k,
            target: b.shape().dims().first().copied().unwrap_or(0),
        }),
    }
}

/// Run the actual MLX matmul on two shape-validated f64 buffers.
fn compute(a: &[f64], b: &[f64], a_dims: &[usize], b_dims: &[usize]) -> Vec<f64> {
    let a_mlx = dense_to_mlx(a, a_dims);
    let b_mlx = dense_to_mlx(b, b_dims);
    let c = a_mlx
        .matmul(&b_mlx)
        .expect("mlx matmul on pre-validated shapes should not fail");
    mlx_to_dense_data(c)
}
