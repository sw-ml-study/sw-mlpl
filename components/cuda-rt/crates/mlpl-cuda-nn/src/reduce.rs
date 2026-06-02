//! CUDA-backed reductions. `mean` and `argmax` run on the GPU
//! (candle `mean`/`argmax`, which drop the reduced axis like the CPU
//! path). `reduce_mul` delegates to the CPU runtime: candle has no
//! `prod` reduction and the op is low-value on the GPU.
//!
//! Axis semantics mirror `mlpl-rt::reductions`: `Option<usize>`
//! (None = flat); a reduced axis drops its label; an out-of-range
//! axis is `IndexOutOfBounds`.

use candle_core::DType;
use mlpl_array::{ArrayError, DenseArray, Shape};
use mlpl_cuda_rt::{Labels, cuda_to_dense_data, dense_to_cuda, finalize};

/// Mean over `axis` (None = flat scalar).
///
/// # Errors
/// `IndexOutOfBounds` if `axis` is out of range.
///
/// # Panics
/// Panics if the candle mean kernel fails on a validated axis.
pub fn mean(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    let t = dense_to_cuda(a.data(), a.shape().dims());
    let Some(ax) = axis else {
        let r = t.mean_all().expect("cuda mean_all");
        return finalize(Shape::new(vec![]), cuda_to_dense_data(&r), None);
    };
    if ax >= a.rank() {
        return Err(ArrayError::IndexOutOfBounds {
            axis: ax,
            index: ax,
            size: a.rank(),
        });
    }
    let r = t.mean(ax).expect("cuda mean over validated axis");
    let (dims, labels) = drop_axis(a, ax);
    finalize(Shape::new(dims), cuda_to_dense_data(&r), labels)
}

/// Index of the maximum over `axis` (None = flat), as f64 to match
/// the CPU path's index representation.
///
/// # Errors
/// `EmptyArray` for a flat argmax of an empty array; `IndexOutOfBounds`
/// if `axis` is out of range.
///
/// # Panics
/// Panics if the candle argmax kernel fails on a validated axis.
pub fn argmax(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    let t = dense_to_cuda(a.data(), a.shape().dims());
    let Some(ax) = axis else {
        if a.elem_count() == 0 {
            return Err(ArrayError::EmptyArray);
        }
        let flat = t.flatten_all().expect("flatten for flat argmax");
        let r = flat.argmax(0).expect("cuda flat argmax");
        let f = r.to_dtype(DType::F32).expect("argmax index to f32");
        return finalize(Shape::new(vec![]), cuda_to_dense_data(&f), None);
    };
    if ax >= a.rank() {
        return Err(ArrayError::IndexOutOfBounds {
            axis: ax,
            index: ax,
            size: a.rank(),
        });
    }
    let r = t.argmax(ax).expect("cuda argmax over validated axis");
    let f = r.to_dtype(DType::F32).expect("argmax index to f32");
    let (dims, labels) = drop_axis(a, ax);
    finalize(Shape::new(dims), cuda_to_dense_data(&f), labels)
}

/// Product reduction over `axis` (None = flat). Delegates to the CPU
/// path -- candle has no `prod` reduction and this is low-value on
/// the GPU (mirrors how `transpose` delegates in `mlpl-cuda-rt`).
///
/// # Errors
/// Propagates `mlpl-rt::reduce_mul`'s axis errors.
pub fn reduce_mul(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    mlpl_rt::reduce_mul(a, axis)
}

/// Drop axis `ax` from `a`'s dims and labels.
fn drop_axis(a: &DenseArray, ax: usize) -> (Vec<usize>, Labels) {
    let mut dims: Vec<usize> = a.shape().dims().to_vec();
    dims.remove(ax);
    let labels = a.labels().map(|lbls| {
        let mut out = lbls.to_vec();
        out.remove(ax);
        out
    });
    (dims, labels)
}
