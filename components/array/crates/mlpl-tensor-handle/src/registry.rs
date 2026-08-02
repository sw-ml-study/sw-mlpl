//! Process-global backend registration -- the same inversion idiom
//! as `mlpl-eval-state::register_gpu_step`: the binary (or a test)
//! registers once at startup, everything below reaches the backend
//! through the registry without linking it.

use std::sync::{Arc, OnceLock};

use mlpl_array::DenseArray;

use crate::device::HandleError;
use crate::handle::TensorHandle;
use crate::ops::DeviceOps;

static DEVICE_OPS: OnceLock<Arc<dyn DeviceOps>> = OnceLock::new();

/// Install the process's device backend. First registration wins;
/// repeat calls (tests, multiple init paths) are no-ops.
pub fn register_device_ops(ops: Arc<dyn DeviceOps>) {
    let _ = DEVICE_OPS.set(ops);
}

/// The registered backend, if any.
#[must_use]
pub fn device_ops() -> Option<&'static Arc<dyn DeviceOps>> {
    DEVICE_OPS.get()
}

/// The registered backend or [`HandleError::NoBackend`].
pub(crate) fn require_ops() -> Result<&'static Arc<dyn DeviceOps>, HandleError> {
    DEVICE_OPS.get().ok_or(HandleError::NoBackend)
}

/// Upload a host array to the registered backend.
///
/// # Errors
/// [`HandleError::NoBackend`] without a registration; backend
/// upload failures pass through.
pub fn upload(a: &DenseArray) -> Result<TensorHandle, HandleError> {
    let dev = require_ops()?.upload(a)?;
    crate::metrics::bump(crate::metrics::SeamEvent::Upload);
    Ok(TensorHandle::Dev(dev))
}
