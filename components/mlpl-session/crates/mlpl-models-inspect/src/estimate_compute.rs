//! Per-step stat-collection + VRAM math for
//! `estimate_train`. Extracted from `estimate.rs` so each
//! file stays under the 4-fn warn line.

use mlpl_env_traits::{HasFrozen, HasStrings, HasVars};
use mlpl_eval_core::model::ModelSpec;

use crate::error::InspectError;
use crate::estimate_walk::{Stats, accumulate_hidden_depth};

pub(crate) const DEFAULT_GFLOPS: f64 = 50.0;
pub(crate) const ACTIVATION_FACTOR: f64 = 4.0;
pub(crate) const DEFAULT_DTYPE_BYTES: f64 = 8.0;

pub(crate) fn collect_stats<E: HasVars + HasFrozen>(
    spec: &ModelSpec,
    env: &E,
) -> Result<Stats, InspectError> {
    let mut stats = Stats::default();
    for name in spec.params() {
        if let Some(arr) = env.get(&name) {
            let size = arr.shape().dims().iter().product::<usize>() as f64;
            stats.params += size;
            if !env.is_frozen(&name) {
                stats.trainable += size;
            }
        }
    }
    accumulate_hidden_depth(spec, env, &mut stats);
    if stats.params == 0.0 {
        return Err(InspectError::NoTrainableParams);
    }
    Ok(stats)
}

pub(crate) fn compute_vram(stats: &Stats, batch: f64, seq: f64, dtype_bytes: f64) -> f64 {
    let weight_bytes = stats.params * dtype_bytes;
    let grad_bytes = stats.trainable * dtype_bytes;
    let adam_bytes = 2.0 * stats.trainable * dtype_bytes;
    let activation_bytes =
        batch * seq * stats.hidden * stats.depth * dtype_bytes * ACTIVATION_FACTOR;
    weight_bytes + grad_bytes + adam_bytes + activation_bytes
}

pub(crate) fn device_gflops<E: HasStrings>(env: &E) -> f64 {
    env.get_string("mlpl_device_throughput_gflops")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_GFLOPS)
}
