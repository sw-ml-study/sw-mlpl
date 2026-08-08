//! Native timing built-in: `clock_ms()` -- a high-resolution
//! MONOTONIC clock for benchmarking (demo-memory's
//! inserts/sec, lookups/sec, latency percentiles). Returns
//! milliseconds elapsed since a process-start epoch, as a
//! scalar; only DIFFERENCES are meaningful. Native + connect
//! only (this crate is the native runtime), like `load` /
//! `llm_call`; not a browser builtin.

use std::sync::OnceLock;
use std::time::Instant;

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

/// Names handled here.
pub(crate) const NAMES: &[&str] = &["clock_ms"];

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Dispatch the timing builtins. `None` if not matched.
pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    if name != "clock_ms" {
        return None;
    }
    Some(clock_ms(args))
}

fn clock_ms(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if !args.is_empty() {
        return Err(RuntimeError::InvalidArgument {
            func: "clock_ms".into(),
            reason: format!("takes no arguments, got {}", args.len()),
        });
    }
    let ms = epoch().elapsed().as_secs_f64() * 1000.0;
    Ok(DenseArray::from_scalar(ms))
}
