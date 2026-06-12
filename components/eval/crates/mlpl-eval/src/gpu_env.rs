//! The narrow `Environment` accessor the GPU compute steps depend on.
//!
//! Split out of `gpu_step` so the optimizer-step trait and the
//! environment-accessor trait each own a file (facade discipline). The
//! impl for `Environment` lives in `env_gpu` -- the orphan rule pins it
//! to the crate that owns `Environment`, so it stays in `mlpl-eval` even
//! after the CUDA/MLX compute moves to sibling crates. See
//! docs/build-and-workspace-plan.md.

use mlpl_array::DenseArray;

/// The narrow slice of the `Environment` the GPU compute needs: read /
/// write a named binding (an adapter weight), and read / write an Adam
/// moment buffer keyed by `(optimizer, param, suffix)`. Keeping this
/// minimal lets the GPU crates depend on a tight interface instead of
/// `Environment`'s internals.
pub(crate) trait GpuEnv {
    fn binding(&self, name: &str) -> Option<&DenseArray>;
    fn set_binding(&mut self, name: String, value: DenseArray);
    fn optim_buffer(&self, opt: &str, param: &str, suffix: &str) -> Option<&DenseArray>;
    fn set_optim_buffer(&mut self, opt: &str, param: &str, suffix: &str, value: DenseArray);
}
