//! CUDA-backed normalization: `softmax` / `log_softmax`. Both run on
//! the GPU via candle-nn ops, preserve shape and labels, and share
//! one body via the `norm!` macro (define-once) -- only the candle op
//! differs.

use mlpl_array::{ArrayError, DenseArray};
use mlpl_cuda_rt::{cuda_to_dense_data, dense_to_cuda, finalize};

macro_rules! norm {
    ($name:ident, $doc:literal, $op:path) => {
        #[doc = $doc]
        ///
        /// # Errors
        /// `IndexOutOfBounds` if `axis` is out of range.
        ///
        /// # Panics
        /// Panics if the candle kernel fails on a validated axis.
        pub fn $name(a: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
            if axis >= a.rank() {
                return Err(ArrayError::IndexOutOfBounds {
                    axis,
                    index: axis,
                    size: a.rank(),
                });
            }
            let t = dense_to_cuda(a.data(), a.shape().dims());
            let r = $op(&t, axis).expect(concat!("cuda ", stringify!($name), " on axis"));
            let labels = a.labels().map(<[Option<String>]>::to_vec);
            finalize(a.shape().clone(), cuda_to_dense_data(&r), labels)
        }
    };
}

norm!(
    softmax,
    "`softmax(a, axis)` on the GPU; shape and labels match the input.",
    candle_nn::ops::softmax
);
norm!(
    log_softmax,
    "`log_softmax(a, axis)` on the GPU; shape and labels match input.",
    candle_nn::ops::log_softmax
);
