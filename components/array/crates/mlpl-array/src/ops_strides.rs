//! Saga 33 step 005: row-major stride computation shared by
//! `transpose`, `reduce_axis`, and `argmax_axis`.

/// Compute strides for row-major layout.
pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}
