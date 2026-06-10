//! Linux telemetry sources for `mlpl-monitor`.
//!
//! Three independent sources -- CPU (`/proc/stat`), RAM
//! (`/proc/meminfo`), GPU (`nvidia-smi`) -- each returning `Option` so a
//! missing source degrades to `null` instead of failing the snapshot.
//! Every source separates a pure `parse_*` (testable on a fixed string)
//! from the thin IO `read`/`sample`/`query` that supplies it.

pub mod cpu;
pub mod gpu;
pub mod mem;
