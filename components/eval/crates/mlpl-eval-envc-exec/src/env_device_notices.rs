//! One-shot device notices: the GPU->CPU fallback warning latch and
//! the per-eval user-visible notice queue, drained at eval
//! boundaries.

use mlpl_eval_env::Environment;

impl EnvDeviceNotices for Environment {
    fn take_device_fallback_warning(&mut self) -> bool {
        if self.mlx_fallback_warned {
            return false;
        }
        self.mlx_fallback_warned = true;
        true
    }

    fn push_notice_once(&mut self, msg: String) {
        if !self.notices.contains(&msg) {
            self.notices.push(msg);
        }
    }

    fn take_notices(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notices)
    }
}

/// Fallback-warning latch + per-eval notice queue.
pub trait EnvDeviceNotices {
    /// Take ownership of the "have we already warned about a GPU
    /// (MLX/CUDA) -> CPU fallback?" flag. Returns `true` the first
    /// time it is called per `Environment`, `false` thereafter, so
    /// callers emit the fallback warning at most once.
    fn take_device_fallback_warning(&mut self) -> bool;
    /// Record a user-visible notice once per eval (deduped by text),
    /// drained at the eval boundary -- e.g. when a GPU device fell
    /// back to CPU for an unsupported model shape.
    fn push_notice_once(&mut self, msg: String);
    /// Drain the collected notices (called at the eval boundary).
    fn take_notices(&mut self) -> Vec<String>;
}
