//! Saga E4 step 001: the `metal` feature is ON, so MLX's default
//! device must be the GPU (previously `accelerate`-only builds ran
//! every "MLX" op on Apple's CPU BLAS). Tolerant on hosts whose
//! runtime lacks a Metal GPU (CI VMs): the test skips instead of
//! failing there, matching the crate's compile-anywhere invariant.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_mlx_rt::{DenseArray, Device, DeviceType, Shape};

#[test]
fn default_device_is_the_gpu_and_computes() {
    let dev = Device::try_default().expect("MLX default device");
    let kind = dev.get_type().expect("device type");
    if !matches!(kind, DeviceType::Gpu) {
        eprintln!("skipping: MLX default device is {kind:?} (no Metal GPU at runtime)");
        return;
    }
    // A real op must execute on that device and agree with exact
    // integer math (small ints are exact in f32).
    let a = DenseArray::new(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = DenseArray::new(Shape::new(vec![2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let out = mlpl_mlx_rt::matmul(&a, &b).expect("gpu matmul");
    assert_eq!(out.data(), &[19.0, 22.0, 43.0, 50.0]);
}
