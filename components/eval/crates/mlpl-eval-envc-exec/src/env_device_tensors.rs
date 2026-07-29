//! Saga 33 step 003: peer-resident tensor-handle methods
//! extracted from `env.rs`. These bind a remote handle as a
//! `Value` so subsequent eval can refer to it by name; the
//! actual tensor data lives on the peer.

use mlpl_eval_env::Environment;
use mlpl_eval_types::Value;

impl EnvDeviceTensors for Environment {
    fn set_device_tensor(&mut self, name: String, value: Value) {
        self.device_tensors.insert(name, value);
    }

    fn get_device_tensor(&self, name: &str) -> Option<&Value> {
        self.device_tensors.get(name)
    }

    fn remove_device_tensor(&mut self, name: &str) {
        self.device_tensors.remove(name);
    }
}

/// Peer-resident tensor handle bindings.
pub trait EnvDeviceTensors {
    fn set_device_tensor(&mut self, name: String, value: Value);
    fn get_device_tensor(&self, name: &str) -> Option<&Value>;
    fn remove_device_tensor(&mut self, name: &str);
}
