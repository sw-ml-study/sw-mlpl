//! Backend-free semantics of the handle seam: these tests compile
//! and run on EVERY target (wasm/Linux/mac) because no backend is
//! ever registered in this binary -- the Dev arm must stay inert
//! with clear errors, and the Cpu arm must pass straight through.

use mlpl_array::{DenseArray, Shape};
use mlpl_tensor_handle::{BinKind, HandleError, TensorHandle, device_ops, upload};

fn host(dims: Vec<usize>, data: Vec<f64>) -> TensorHandle {
    TensorHandle::from(DenseArray::new(Shape::new(dims), data).unwrap())
}

#[test]
fn cpu_arm_passes_through() {
    let h = host(vec![2, 3], (0..6).map(f64::from).collect());
    assert_eq!(h.dims(), vec![2, 3]);
    assert!(!h.is_dev());
    assert_eq!(h.to_dense().data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn without_a_backend_everything_is_inert_and_loud() {
    assert!(
        device_ops().is_none(),
        "this test binary must stay backend-free"
    );
    let a = DenseArray::from_scalar(1.0);
    assert_eq!(upload(&a).unwrap_err(), HandleError::NoBackend);
    let (x, y) = (host(vec![2], vec![1.0, 2.0]), host(vec![2], vec![3.0, 4.0]));
    // Two host operands: the CPU path's job, never a silent upload.
    assert_eq!(
        x.dev_binary(BinKind::Add, &y).unwrap_err(),
        HandleError::NotResident
    );
}
