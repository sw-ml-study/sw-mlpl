//! MLX-backed shape primitives (Saga 14 step 002).
//!
//! Shape-metadata ops -- `reshape` and `transpose` -- dispatch
//! through MLX's own shape kernels so the Saga 14 promise "every
//! primitive runs on MLX" holds end-to-end. Values travel through
//! the fp32 round trip in `common::dense_to_mlx` + `mlx_to_dense_data`
//! like every other op, so parity tests share the same tolerance
//! budget.
//!
//! Label propagation mirrors `mlpl-array`:
//! - `reshape` drops labels (a reshaped array no longer carries
//!   the old per-axis identity).
//! - `transpose` reverses the label list alongside the axes.

use mlpl_array::{ArrayError, DenseArray, Shape};

use crate::common::{dense_to_mlx, finalize, mlx_to_dense_data};

/// `reshape(a, dims)` -- reinterpret `a` with new dims. Element
/// count must match or `ArrayError::ShapeMismatch` is returned.
/// Labels drop, matching the CPU path.
pub fn reshape(a: &DenseArray, dims: &[usize]) -> Result<DenseArray, ArrayError> {
    let source = a.elem_count();
    let target: usize = dims.iter().product();
    if source != target {
        return Err(ArrayError::ShapeMismatch { source, target });
    }
    let mlx = dense_to_mlx(a.data(), a.shape().dims());
    let new_shape_i32: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
    let reshaped = mlx
        .reshape(&new_shape_i32)
        .expect("mlx reshape on validated element count");
    finalize(Shape::new(dims.to_vec()), mlx_to_dense_data(reshaped), None)
}

/// `transpose(a)` -- reverse axis order and reorder data to
/// row-major. Labels reverse alongside axes. Infallible, matching
/// `mlpl_rt::transpose`.
///
/// Implementation note: MLX 0.25.3 exposes `transpose` as a lazy
/// view with permuted strides, and `as_slice` on a view still
/// returns the underlying row-major buffer -- mlx-rs does not
/// surface the C `mlx_contiguous` helper that would force a
/// rematerialized copy. We therefore delegate to the CPU path,
/// which is a fixed-size stride walk and cheap at Saga 14 sizes.
/// Once `device("mlx") { }` chains fuse transpose with downstream
/// ops (Phase 2, step 005), the intermediate view never lands in
/// a flat buffer and this fallback disappears.
#[must_use]
pub fn transpose(a: &DenseArray) -> DenseArray {
    a.transpose()
}
