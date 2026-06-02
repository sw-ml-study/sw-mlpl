//! CUDA-backed matrix multiply. Shapes are validated on the Rust
//! side so candle sees only legal inputs; label propagation mirrors
//! `mlpl-array` (contraction labels must agree; survivors are
//! `[lhs.labels[0], rhs.labels[1]]`).

use crate::convert::{Labels, cuda_to_dense_data, dense_to_cuda, finalize};
use mlpl_array::{ArrayError, DenseArray, Shape};

/// Matrix multiply backed by candle. Shape rules match the CPU path:
/// `[m, k] @ [k, n] -> [m, n]` and `[m, k] @ [k] -> [m]`.
///
/// # Errors
/// `RankMismatch` if `lhs` is not rank 2; `ShapeMismatch` if `rhs`'s
/// leading dim does not match `lhs`'s `k`; `LabelMismatch` on
/// disagreeing contraction labels.
pub fn matmul(lhs: &DenseArray, rhs: &DenseArray) -> Result<DenseArray, ArrayError> {
    let [m, k] = lhs.shape().dims() else {
        return Err(ArrayError::RankMismatch {
            expected: 2,
            got: lhs.rank(),
        });
    };
    let (m, k) = (*m, *k);
    let labels = matmul_labels(lhs, rhs)?;
    match rhs.shape().dims() {
        [k2, n] if *k2 == k => {
            let data = compute(lhs.data(), rhs.data(), &[m, k], &[k, *n]);
            finalize(Shape::new(vec![m, *n]), data, labels)
        }
        [k2] if *k2 == k => {
            let data = compute(lhs.data(), rhs.data(), &[m, k], &[k]);
            finalize(Shape::vector(m), data, labels)
        }
        _ => Err(ArrayError::ShapeMismatch {
            source: k,
            target: rhs.shape().dims().first().copied().unwrap_or(0),
        }),
    }
}

/// Run the candle matmul on two shape-validated f64 buffers. candle
/// needs 2D operands, so a matrix-vector product reshapes the
/// vector to `[k, 1]` and flattens the `[m, 1]` result back to `[m]`.
fn compute(a: &[f64], b: &[f64], a_dims: &[usize], b_dims: &[usize]) -> Vec<f64> {
    let at = dense_to_cuda(a, a_dims);
    let bt = dense_to_cuda(b, b_dims);
    let c = if b_dims.len() == 1 {
        let b2 = bt.reshape((b_dims[0], 1)).expect("reshape vec operand");
        let prod = at.matmul(&b2).expect("cuda matmul (matrix-vector)");
        prod.reshape((a_dims[0],)).expect("flatten matmul result")
    } else {
        at.matmul(&bt).expect("cuda matmul (matrix-matrix)")
    };
    cuda_to_dense_data(&c)
}

/// Mirror of `mlpl-array`'s private `matmul_labels`.
fn matmul_labels(a: &DenseArray, b: &DenseArray) -> Result<Labels, ArrayError> {
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
