//! Saga 33 step 003: device-stack + MLX-fallback-warning
//! methods extracted from `env.rs`. The `device("...")` block
//! pushes a name onto `device_stack` on entry and pops on exit;
//! `device()` reads the innermost entry. The MLX fallback
//! warning is a per-Environment latch (warn at most once).

use crate::env::Environment;

impl Environment {
    /// Current active device target (Saga 14 step 004). Returns
    /// `"cpu"` when no `device("...")` block is in scope.
    #[must_use]
    pub fn device(&self) -> &str {
        self.device_stack.last().map_or("cpu", String::as_str)
    }

    /// Push a new device target onto the stack. Called on
    /// `device("...") { ... }` entry.
    pub fn push_device(&mut self, target: String) {
        self.device_stack.push(target);
    }

    /// Pop the innermost device target. Called on `device(...)`
    /// block exit. No-op when the stack is empty (defensive).
    pub fn pop_device(&mut self) {
        self.device_stack.pop();
    }

    /// Take ownership of the "have we already warned about a GPU
    /// (MLX/CUDA) -> CPU fallback?" flag. Returns `true` the first
    /// time it is called per `Environment`, `false` thereafter, so
    /// callers emit the fallback warning at most once. A build is
    /// only ever one GPU backend, so a single latch suffices.
    pub fn take_device_fallback_warning(&mut self) -> bool {
        if self.mlx_fallback_warned {
            return false;
        }
        self.mlx_fallback_warned = true;
        true
    }
}
