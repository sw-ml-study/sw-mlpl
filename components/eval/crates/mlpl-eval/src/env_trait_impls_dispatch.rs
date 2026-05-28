//! Saga 33 step 015: `impl HasDispatch for Environment`.
//! Delegates to `crate::device::dispatched_call` and collapses
//! the rich `EvalError` into a `DispatchError` so sub-crates
//! can convert via their own error types.

use mlpl_array::DenseArray;
use mlpl_env_traits::{DispatchError, HasDispatch};

use crate::env::Environment;
use mlpl_eval_types::EvalError;

impl HasDispatch for Environment {
    fn dispatch(&self, op: &str, args: Vec<DenseArray>) -> Result<DenseArray, DispatchError> {
        crate::device::dispatched_call(self, op, args).map_err(eval_to_dispatch)
    }
}

fn eval_to_dispatch(e: EvalError) -> DispatchError {
    match e {
        EvalError::ArrayError(a) => DispatchError::ArrayError(a),
        EvalError::Unsupported(s) => DispatchError::UnknownOp(s),
        other => DispatchError::Runtime(format!("{other}")),
    }
}
