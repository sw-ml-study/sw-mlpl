//! Shared CUDA plumbing: the process-wide device handle, the
//! f64 <-> f32 conversion boundary, and the `DenseArray` finalize
//! dance. Every CUDA op (here and in sibling crates) routes through
//! these so the GPU round trip lives in one place.

use candle_core::{DType, Device, Tensor};
use mlpl_array::{ArrayError, DenseArray, Shape};
use std::sync::OnceLock;

/// Optional per-axis labels carried alongside an array. Aliased so
/// op signatures stay readable (and clippy-`type_complexity` clean).
pub type Labels = Option<Vec<Option<String>>>;

static CUDA: OnceLock<Device> = OnceLock::new();

/// Borrow the process-wide CUDA device 0, initializing it on first
/// use (the handle is cheap to clone -- an `Arc` inside).
///
/// # Panics
/// Panics if the candle CUDA backend cannot open device 0. Callers
/// reach this only under the `cuda` feature on Linux/`x86_64`, where
/// the dispatch layer guarantees a GPU is present.
pub fn cuda_device() -> &'static Device {
    CUDA.get_or_init(|| Device::new_cuda(0).expect("CUDA device 0 initializes"))
}

/// Build a contiguous fp32 CUDA tensor from a `DenseArray`'s f64
/// buffer; candle performs the f64 -> f32 downcast via `to_dtype`.
///
/// # Panics
/// Panics if candle cannot allocate or cast the tensor for a
/// pre-validated element count.
#[must_use]
pub fn dense_to_cuda(data: &[f64], dims: &[usize]) -> Tensor {
    Tensor::from_vec(data.to_vec(), dims.to_vec(), cuda_device())
        .expect("cuda tensor from pre-validated shape")
        .to_dtype(DType::F32)
        .expect("cast f64 tensor to f32 for GPU compute")
}

/// Materialize a CUDA tensor and cast its flat contents back to f64.
///
/// # Panics
/// Panics if candle cannot copy the tensor to host memory.
#[must_use]
pub fn cuda_to_dense_data(t: &Tensor) -> Vec<f64> {
    let flat = t.flatten_all().expect("flatten cuda tensor");
    let out: Vec<f32> = flat.to_vec1::<f32>().expect("cuda tensor to host f32");
    out.iter().map(|&x| f64::from(x)).collect()
}

/// Wrap `data` in a `DenseArray` of `shape`, attaching `labels` if
/// present. Consolidates the new + `with_labels` dance every op does.
///
/// # Errors
/// Returns `ArrayError` if the data length does not match `shape`
/// or the labels are invalid for the shape.
pub fn finalize(shape: Shape, data: Vec<f64>, labels: Labels) -> Result<DenseArray, ArrayError> {
    let array = DenseArray::new(shape, data)?;
    match labels {
        Some(lbls) => array.with_labels(lbls),
        None => Ok(array),
    }
}
