//! Environment EXECUTION-CONTEXT capabilities as traits (eval
//! decomposition capability peel): the device stack, device
//! notices, peer-tensor handles, per-tensor placement, the peer
//! dispatcher, and interrupt checkpoints.

pub mod env_device;
pub mod env_device_notices;
pub mod env_device_tensors;
pub mod env_interrupt;
pub mod env_peer;
pub mod env_tensor_device;

pub use env_device::EnvDevice;
pub use env_device_notices::EnvDeviceNotices;
pub use env_device_tensors::EnvDeviceTensors;
pub use env_interrupt::EnvInterrupt;
pub use env_peer::EnvPeer;
pub use env_tensor_device::EnvTensorDevice;
