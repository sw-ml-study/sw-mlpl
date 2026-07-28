//! The narrow `Environment` accessor the GPU compute steps depend
//! on. The impl for `Environment` stays with the crate that owns
//! `Environment` (orphan rule); the GPU compute crates depend on
//! this tight interface instead of `Environment`'s internals.

use mlpl_array::DenseArray;

/// The narrow slice of the `Environment` the GPU compute needs: read /
/// write a named binding (an adapter weight), and read / write an Adam
/// moment buffer keyed by `(optimizer, param, suffix)`.
pub trait GpuEnv {
    fn binding(&self, name: &str) -> Option<&DenseArray>;
    fn set_binding(&mut self, name: String, value: DenseArray);
    fn optim_buffer(&self, opt: &str, param: &str, suffix: &str) -> Option<&DenseArray>;
    fn set_optim_buffer(&mut self, opt: &str, param: &str, suffix: &str, value: DenseArray);
}
