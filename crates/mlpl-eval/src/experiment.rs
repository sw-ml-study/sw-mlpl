//! `experiment "name" { body }` evaluator + record type
//! (Saga 12 step 007).

use std::collections::BTreeMap;
use std::path::Path;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::{Trace, TraceValue};
use serde::Serialize;

use crate::env::Environment;
use crate::error::EvalError;

/// One recorded run. Written to `<exp_dir>/<name>/<ts>/run.json`
/// by the terminal REPL; also appended to `env.experiment_log`
/// so the web REPL can surface runs via `:experiments`.
#[derive(Clone, Debug, Serialize)]
pub struct ExperimentRecord {
    /// Name passed to the `experiment "..."` form.
    pub name: String,
    /// Wall-clock `SystemTime::duration_since(UNIX_EPOCH)` in
    /// nanoseconds at run-entry. Used to make the on-disk
    /// timestamp subdir unique.
    pub timestamp_ns: u128,
    /// `_metric`-suffixed scalar values captured at run-exit.
    pub metrics: BTreeMap<String, f64>,
    /// Shape metadata for every bound tracked parameter at
    /// run-exit. Keyed by param name.
    pub params_snapshot: BTreeMap<String, ParamShape>,
}

/// Shape snapshot stored inside an `ExperimentRecord`.
#[derive(Clone, Debug, Serialize)]
pub struct ParamShape {
    /// Positional dims.
    pub shape: Vec<usize>,
    /// Per-axis labels when the array is labeled; `None` when
    /// the array has no labels.
    pub labels: Option<Vec<Option<String>>>,
}

/// Evaluate an `experiment "name" { body }` block.
pub(crate) fn eval_experiment(
    name: &str,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let timestamp_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut last = DenseArray::from_scalar(0.0);
    for stmt in body {
        last = crate::eval::eval_expr(stmt, env, trace)?.into_array()?;
    }
    let metrics = collect_metrics(env);
    let params_snapshot = collect_param_shapes(env);
    let record = ExperimentRecord {
        name: name.to_string(),
        timestamp_ns,
        metrics,
        params_snapshot,
    };
    if let Some(dir) = env.exp_dir().cloned() {
        write_record_to_disk(&dir, &record)
            .map_err(|e| EvalError::Unsupported(format!("experiment: {e}")))?;
    }
    env.push_experiment_log(record);
    Ok(("experiment", vec![], last))
}

fn collect_metrics(env: &Environment) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for (name, arr) in env.vars_iter() {
        if name.ends_with("_metric") && arr.rank() == 0 {
            out.insert(name.clone(), arr.data()[0]);
        }
    }
    out
}

fn collect_param_shapes(env: &Environment) -> BTreeMap<String, ParamShape> {
    let mut out = BTreeMap::new();
    for (name, arr) in env.params() {
        out.insert(
            name.clone(),
            ParamShape {
                shape: arr.shape().dims().to_vec(),
                labels: arr.labels().map(<[_]>::to_vec),
            },
        );
    }
    out
}

fn write_record_to_disk(dir: &Path, rec: &ExperimentRecord) -> Result<(), String> {
    let run_dir = dir.join(&rec.name).join(rec.timestamp_ns.to_string());
    std::fs::create_dir_all(&run_dir)
        .map_err(|e| format!("creating {}: {e}", run_dir.display()))?;
    let json = serde_json::to_string_pretty(rec).map_err(|e| format!("serializing record: {e}"))?;
    std::fs::write(run_dir.join("run.json"), json).map_err(|e| format!("writing run.json: {e}"))?;
    Ok(())
}
