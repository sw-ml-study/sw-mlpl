//! Unary elementwise maps on the GPU: negation and the activations
//! (`exp`/`log`/`relu`/`sigmoid`/`tanh`). Each is a 1:1 map, so shape
//! and labels pass through unchanged; the six share one body via the
//! `unary!` macro (define-once, per loose-coupling.md) -- only the
//! candle op differs.

use candle_core::Tensor;
use mlpl_array::DenseArray;
use mlpl_cuda_rt::{cuda_to_dense_data, dense_to_cuda};

/// Apply a candle unary op to `a` on the GPU, preserving shape and
/// labels. The shared body behind every unary op.
///
/// # Panics
/// Panics if `DenseArray` reconstruction fails on data the input
/// already carried (cannot happen for a 1:1 map).
fn apply_unary(a: &DenseArray, op: impl FnOnce(&Tensor) -> Tensor) -> DenseArray {
    let r = op(&dense_to_cuda(a.data(), a.shape().dims()));
    let array = DenseArray::new(a.shape().clone(), cuda_to_dense_data(&r))
        .expect("output element count matches input");
    match a.labels() {
        Some(lbls) => array
            .with_labels(lbls.to_vec())
            .expect("labels already validated on input"),
        None => array,
    }
}

macro_rules! unary {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Panics
        /// Panics if the candle kernel fails on the pre-validated input.
        #[must_use]
        pub fn $name(a: &DenseArray) -> DenseArray {
            apply_unary(a, $op)
        }
    };
}

unary!(neg, "Unary elementwise negation.", |t: &Tensor| t
    .affine(-1.0, 0.0)
    .expect("cuda neg on validated shape"));
unary!(exp, "`exp(x)` elementwise.", |t: &Tensor| t
    .exp()
    .expect("cuda exp on validated shape"));
unary!(log, "Natural `log(x)` elementwise.", |t: &Tensor| t
    .log()
    .expect("cuda log on validated shape"));
unary!(relu, "`relu(x)` = max(x, 0) elementwise.", |t: &Tensor| t
    .relu()
    .expect("cuda relu on validated shape"));
unary!(
    sigmoid,
    "Logistic `sigmoid(x)` elementwise.",
    |t: &Tensor| { candle_nn::ops::sigmoid(t).expect("cuda sigmoid on validated shape") }
);
unary!(tanh, "`tanh(x)` elementwise.", |t: &Tensor| t
    .tanh()
    .expect("cuda tanh on validated shape"));
