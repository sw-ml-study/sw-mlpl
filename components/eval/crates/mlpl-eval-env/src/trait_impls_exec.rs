//! `mlpl-env-traits` EXECUTION impls for `Environment`:
//! `HasTensorDevices` and `HasDispatch` (dispatch routes through the
//! hub-installed hook in `dispatch_hook`).

use mlpl_array::DenseArray;
use mlpl_env_traits::{DispatchError, HasDispatch};
use mlpl_env_traits::{HasModelIds, HasTensorDevices};
use mlpl_eval_types::EvalError;

use crate::env::Environment;

impl HasTensorDevices for Environment {
    fn tensor_device(&self, name: &str) -> &str {
        self.tensor_device.get(name).map_or("cpu", String::as_str)
    }
    fn set_tensor_device(&mut self, name: String, target: String) {
        self.tensor_device.insert(name, target);
    }
}

impl HasModelIds for Environment {
    fn alloc_model_id(&mut self) -> u64 {
        let id = self.next_model_id;
        self.next_model_id += 1;
        id
    }
}

impl HasDispatch for Environment {
    fn dispatch(&self, op: &str, args: Vec<DenseArray>) -> Result<DenseArray, DispatchError> {
        crate::dispatch_hook::dispatch_or_err(self, op, args).map_err(eval_to_dispatch)
    }
}

fn eval_to_dispatch(e: EvalError) -> DispatchError {
    match e {
        EvalError::ArrayError(a) => DispatchError::ArrayError(a),
        EvalError::Unsupported(s) => DispatchError::UnknownOp(s),
        other => DispatchError::Runtime(format!("{other}")),
    }
}
