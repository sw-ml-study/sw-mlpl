//! Convolution built-in functions extracted from mlpl-runtime.

mod conv2d;
mod pool2d;

pub const NAMES: &[&str] = &["conv2d", "pool2d"];

pub fn try_call(
    name: &str,
    args: Vec<mlpl_array::DenseArray>,
) -> Option<Result<mlpl_array::DenseArray, mlpl_runtime_core::error::RuntimeError>> {
    match name {
        "conv2d" => Some(conv2d::run(args)),
        "pool2d" => Some(pool2d::run(args)),
        _ => None,
    }
}
