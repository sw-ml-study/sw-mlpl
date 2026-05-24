//! Saga 33 step 012: `impl HasTensorDevices + HasModelIds for
//! Environment`. The tensor-device methods delegate to env_tensor_device.rs;
//! the model-id allocator wraps the `pub(crate) next_model_id`
//! field (auto-increment).

use mlpl_env_traits::{HasModelIds, HasTensorDevices};

use crate::env::Environment;

impl HasTensorDevices for Environment {
    fn tensor_device(&self, name: &str) -> &str {
        Environment::tensor_device(self, name)
    }
    fn set_tensor_device(&mut self, name: String, target: String) {
        Environment::set_tensor_device(self, name, target);
    }
}

impl HasModelIds for Environment {
    fn alloc_model_id(&mut self) -> u64 {
        let id = self.next_model_id;
        self.next_model_id += 1;
        id
    }
}
