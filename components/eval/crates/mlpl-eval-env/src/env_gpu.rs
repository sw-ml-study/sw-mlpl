//! The shared `GpuEnv` accessor impl for `Environment` -- read/write
//! bindings + Adam moment buffers, nothing else.
//!
//! Backend-agnostic: it references no CUDA/MLX type, so CUDA and MLX
//! share this one impl (cfg any-gpu). It lives here rather than in
//! `env.rs` to keep that module under its function-count budget, and it
//! stays in `mlpl-eval` permanently -- the orphan rule pins
//! `impl GpuEnv for Environment` to the crate that owns `Environment`.

use crate::env::Environment;
use mlpl_array::DenseArray;
use mlpl_eval_state::GpuEnv;

impl GpuEnv for Environment {
    fn binding(&self, name: &str) -> Option<&DenseArray> {
        self.vars.get(name)
    }
    fn set_binding(&mut self, name: String, value: DenseArray) {
        self.vars.insert(name, value);
    }
    fn optim_buffer(&self, opt: &str, param: &str, suffix: &str) -> Option<&DenseArray> {
        self.optim_state
            .buffers
            .get(&(opt.to_string(), param.to_string(), suffix.to_string()))
    }
    fn set_optim_buffer(&mut self, opt: &str, param: &str, suffix: &str, value: DenseArray) {
        self.optim_state.buffers.insert(
            (opt.to_string(), param.to_string(), suffix.to_string()),
            value,
        );
    }
}
