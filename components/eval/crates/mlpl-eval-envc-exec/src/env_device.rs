//! Device-stack accessors: the `device("...")` block pushes a name
//! onto `device_stack` on entry and pops on exit; `device()` reads
//! the innermost entry.

use mlpl_eval_env::Environment;

impl EnvDevice for Environment {
    fn device(&self) -> &str {
        self.device_stack.last().map_or("cpu", String::as_str)
    }

    fn push_device(&mut self, target: String) {
        self.device_stack.push(target);
    }

    fn pop_device(&mut self) {
        self.device_stack.pop();
    }
}

/// The active-device stack behind `device("...") { ... }` blocks.
pub trait EnvDevice {
    /// Current active device target (Saga 14 step 004). Returns
    /// `"cpu"` when no `device("...")` block is in scope.
    fn device(&self) -> &str;
    /// Push a new device target onto the stack. Called on
    /// `device("...") { ... }` entry.
    fn push_device(&mut self, target: String);
    /// Pop the innermost device target. Called on `device(...)`
    /// block exit. No-op when the stack is empty (defensive).
    fn pop_device(&mut self);
}
