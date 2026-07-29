//! Saga 33 step 003: per-tensor device-placement methods
//! extracted from `env.rs`. `to_device(x, target)` calls
//! `set_tensor_device`; `apply(model, X)` and similar
//! placement-sensitive ops read `tensor_device`.

use mlpl_eval_env::Environment;

impl EnvTensorDevice for Environment {
    fn tensor_device(&self, name: &str) -> &str {
        self.tensor_device.get(name).map_or("cpu", String::as_str)
    }

    fn set_tensor_device(&mut self, name: String, target: String) {
        self.tensor_device.insert(name, target);
    }
}

/// Per-tensor device placement stamps.
pub trait EnvTensorDevice {
    /// Device placement recorded for the tensor bound to `name`.
    /// Saga 14 step 005. Returns `"cpu"` for names that were never
    /// passed through `to_device` and for names that don't exist
    /// (the evaluator looks up the value separately via `get`).
    fn tensor_device(&self, name: &str) -> &str;
    /// Stamp `name` with device placement `target`. Saga 14 step
    /// 005; used by `to_device(x, target)` and by model
    /// constructors when they allocate params inside a
    /// `device("mlx") { }` block.
    fn set_tensor_device(&mut self, name: String, target: String);
}
