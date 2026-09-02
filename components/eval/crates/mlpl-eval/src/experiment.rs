//! `experiment "name" { body }` evaluator + record type
//! (Saga 12 step 007) and the registry readers used by
//! `:experiments` / `compare()` (Saga 12 step 008).

use crate::env_api::*;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::{Trace, TraceValue};

use crate::env::Environment;
use mlpl_eval_types::EvalError;

// The record types moved to mlpl-eval-state (env-types-out step);
// re-exported so `crate::experiment::ExperimentRecord` keeps working.
pub use mlpl_eval_state::{ExperimentRecord, ParamShape};

/// Evaluate an `experiment "name" { body }` block.
pub(crate) fn eval_experiment(
    name: &str,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let timestamp_ns = experiment_timestamp_ns();
    let mut last = DenseArray::from_scalar(0.0);
    for stmt in body {
        last = crate::eval::eval_expr(stmt, env, trace)?.into_array()?;
    }
    let metrics = crate::experiment_store::collect_metrics(env);
    let params_snapshot = crate::experiment_store::collect_param_shapes(env);
    let record = ExperimentRecord {
        name: name.to_string(),
        timestamp_ns,
        metrics,
        params_snapshot,
    };
    if let Some(dir) = env.exp_dir().cloned() {
        crate::experiment_store::write_record_to_disk(&dir, &record)
            .map_err(|e| EvalError::Unsupported(format!("experiment: {e}")))?;
    }
    env.push_experiment_log(record);
    Ok(("experiment", vec![], last))
}

/// Per-run timestamp used to make on-disk run dirs unique and to
/// sort the registry. `wasm32-unknown-unknown` has no real clock
/// (SystemTime::now() panics), so on that target we fall back to a
/// monotonic in-process counter, which is sufficient for ordering
/// runs captured in `env.experiment_log` even though the resulting
/// value is not wall-clock time.
#[cfg(not(target_arch = "wasm32"))]
fn experiment_timestamp_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn experiment_timestamp_ns() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed) as u128
}

/// Produce the `:experiments` REPL output: merges
/// `env.experiment_log` (memory) with any on-disk records under
/// `env.exp_dir`, sorts by `timestamp_ns`, and prints one line per
/// run with name, timestamp, and top-line metric. Saga 12 step 008.
pub fn format_registry(env: &crate::env::Environment) -> String {
    let mut all: Vec<ExperimentRecord> = env.experiment_log().to_vec();
    if let Some(dir) = env.exp_dir() {
        all.extend(crate::experiment_store::read_records_from_disk(dir));
    }
    if all.is_empty() {
        return "(no experiments recorded)".into();
    }
    all.sort_by_key(|r| r.timestamp_ns);
    let mut out = String::new();
    for r in &all {
        let summary = r
            .metrics
            .iter()
            .next()
            .map_or("(no metrics)".to_string(), |(k, v)| format!("{k}={v}"));
        out.push_str(&format!("  {} @ {} -- {summary}\n", r.name, r.timestamp_ns));
    }
    out.truncate(out.trim_end().len());
    out
}
