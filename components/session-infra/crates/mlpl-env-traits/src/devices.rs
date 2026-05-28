//! `HasTensorDevices` + `HasModelIds`: per-tensor device
//! placement plus the model-id allocator. Bundled into one
//! file so the trait crate stays under the 7-module FAIL
//! line.

pub trait HasTensorDevices {
    /// Device placement recorded for the tensor bound to `name`.
    /// Returns `"cpu"` for names that were never stamped.
    fn tensor_device(&self, name: &str) -> &str;
    /// Stamp `name` with device placement `target`.
    fn set_tensor_device(&mut self, name: String, target: String);
}

/// Monotonic counter that mints fresh layer ids for model
/// constructors and `clone_model` (so cloned layers don't
/// collide with existing names).
pub trait HasModelIds {
    /// Returns the current id and increments the counter.
    fn alloc_model_id(&mut self) -> u64;
}
