//! macOS telemetry sources for `mlpl-monitor`, all without sudo.
//!
//! Three independent sources -- CPU (Mach `host_statistics64`), RAM
//! (`vm_stat` + `sysctl hw.memsize`), GPU (`ioreg -c IOAccelerator`) --
//! each returning `Option`/empty so a missing source degrades to `null`
//! instead of failing the snapshot. Every source separates a pure
//! `parse_*` (testable on a fixed string) from the thin IO that supplies
//! it. The Apple analog of `mlpl-monitor-linux`.

pub mod cpu;
pub mod gpu;
pub mod host;
pub mod mem;
