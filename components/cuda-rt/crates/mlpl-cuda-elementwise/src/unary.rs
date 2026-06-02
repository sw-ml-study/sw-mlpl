//! Unary elementwise negation on the GPU.

use mlpl_array::DenseArray;
use mlpl_cuda_rt::{cuda_to_dense_data, dense_to_cuda};

/// Unary elementwise negation. Infallible: shape and labels pass
/// through from `a`.
///
/// # Panics
/// Panics if the candle kernel or the f64 round trip fails on the
/// pre-validated input.
#[must_use]
pub fn neg(a: &DenseArray) -> DenseArray {
    let t = dense_to_cuda(a.data(), a.shape().dims());
    let r = t.affine(-1.0, 0.0).expect("cuda neg on validated shape");
    let array = DenseArray::new(a.shape().clone(), cuda_to_dense_data(&r))
        .expect("output element count matches input");
    match a.labels() {
        Some(lbls) => array
            .with_labels(lbls.to_vec())
            .expect("labels already validated on input"),
        None => array,
    }
}
