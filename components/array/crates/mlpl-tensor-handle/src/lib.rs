//! Device-resident tensor seam (saga E4 step 002).
//!
//! [`TensorHandle`] is either a host `DenseArray` (f64, the
//! bit-exact CPU reference) or an opaque [`DeviceArray`] living on
//! a compute backend. Ops on resident handles route through the
//! process-global [`DeviceOps`] registry; downloading (`to_dense`)
//! is the ONLY point where a lazy backend graph is forced. No GPU
//! library is linked here -- backends register themselves at
//! startup (the same inversion as `mlpl-eval-state`'s `gpu_step`).

mod device;
mod handle;
mod handle_ops;
pub mod metrics;
mod ops;
mod registry;

pub use device::{AxisKind, BinKind, Dev, DeviceArray, HandleError, UnaryKind};
pub use handle::TensorHandle;
pub use metrics::{SeamEvent, bump, bump_if, seam_reset, seam_snapshot};
pub use ops::DeviceOps;
pub use registry::{device_ops, register_device_ops, upload};
