//! Saga 33 step 003: peer-resident tensor-handle methods
//! extracted from `env.rs`. These bind a remote handle as a
//! `Value` so subsequent eval can refer to it by name; the
//! actual tensor data lives on the peer.

use crate::env::Environment;
use mlpl_eval_types::Value;

impl Environment {
    pub fn set_device_tensor(&mut self, name: String, value: Value) {
        self.device_tensors.insert(name, value);
    }

    #[must_use]
    pub fn get_device_tensor(&self, name: &str) -> Option<&Value> {
        self.device_tensors.get(name)
    }

    pub fn remove_device_tensor(&mut self, name: &str) {
        self.device_tensors.remove(name);
    }
}
