//! `experiment "name" { body }` evaluator + record type
//! (Saga 12 step 007) and the registry readers used by
//! `:experiments` / `compare()` (Saga 12 step 008).

use std::collections::BTreeMap;
use std::path::Path;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::{Trace, TraceValue};
use serde::{Deserialize, Serialize};

use crate::env::Environment;
use crate::error::EvalError;

/// One recorded run. Written to `<exp_dir>/<name>/<ts>/run.json`
/// by the terminal REPL; also appended to `env.experiment_log`
/// so the web REPL can surface runs via `:experiments`.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    let timestamp_ns = experiment_timestamp_ns();
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

/// Walk `<exp_dir>/*/*/run.json` and return every record that
/// deserializes cleanly. Malformed `run.json` files are skipped
/// silently -- a future step can wire up a warning channel.
pub(crate) fn read_records_from_disk(dir: &Path) -> Vec<ExperimentRecord> {
    let mut out = Vec::new();
    let Ok(name_dirs) = std::fs::read_dir(dir) else {
        return out;
    };
    for name_entry in name_dirs.flatten() {
        let Ok(ts_dirs) = std::fs::read_dir(name_entry.path()) else {
            continue;
        };
        for ts_entry in ts_dirs.flatten() {
            let run_json = ts_entry.path().join("run.json");
            let Ok(body) = std::fs::read_to_string(&run_json) else {
                continue;
            };
            if let Ok(rec) = serde_json::from_str::<ExperimentRecord>(&body) {
                out.push(rec);
            }
        }
    }
    out
}

/// Produce the `:experiments` REPL output: merges
/// `env.experiment_log` (memory) with any on-disk records under
/// `env.exp_dir`, sorts by `timestamp_ns`, and prints one line per
/// run with name, timestamp, and top-line metric. Saga 12 step 008.
pub fn format_registry(env: &crate::env::Environment) -> String {
    let mut all: Vec<ExperimentRecord> = env.experiment_log().to_vec();
    if let Some(dir) = env.exp_dir() {
        all.extend(read_records_from_disk(dir));
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

/// `compare(name_a, name_b)` builtin dispatch. Returns a
/// `Value::Str` containing a side-by-side of the most recent
/// runs with each name. Errors if either name has no records or
/// args are malformed.
pub(crate) fn dispatch_compare(
    args: &[Expr],
    env: &mut crate::env::Environment,
) -> Result<crate::value::Value, crate::error::EvalError> {
    if args.len() != 2 {
        return Err(crate::error::EvalError::BadArity {
            func: "compare".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let Expr::StrLit(a, _) = &args[0] else {
        return Err(crate::error::EvalError::Unsupported(
            "compare: arguments must be string literals".into(),
        ));
    };
    let Expr::StrLit(b, _) = &args[1] else {
        return Err(crate::error::EvalError::Unsupported(
            "compare: arguments must be string literals".into(),
        ));
    };
    let ra = latest_by_name(env, a).ok_or_else(|| {
        crate::error::EvalError::Unsupported(format!("compare: no run named {a:?}"))
    })?;
    let rb = latest_by_name(env, b).ok_or_else(|| {
        crate::error::EvalError::Unsupported(format!("compare: no run named {b:?}"))
    })?;
    Ok(crate::value::Value::Str(render_compare(&ra, &rb)))
}

fn latest_by_name(env: &crate::env::Environment, name: &str) -> Option<ExperimentRecord> {
    let mut all: Vec<ExperimentRecord> = env.experiment_log().to_vec();
    if let Some(dir) = env.exp_dir() {
        all.extend(read_records_from_disk(dir));
    }
    all.into_iter()
        .filter(|r| r.name == name)
        .max_by_key(|r| r.timestamp_ns)
}

fn render_compare(a: &ExperimentRecord, b: &ExperimentRecord) -> String {
    let mut keys: std::collections::BTreeSet<String> = a.metrics.keys().cloned().collect();
    keys.extend(b.metrics.keys().cloned());
    if keys.is_empty() {
        return format!(
            "compare {} vs {} -- (no metrics on either run)",
            a.name, b.name
        );
    }
    let mut out = format!("compare {} vs {}\n", a.name, b.name);
    for k in &keys {
        let av = a.metrics.get(k).copied();
        let bv = b.metrics.get(k).copied();
        let delta = match (av, bv) {
            (Some(x), Some(y)) => format!(" (delta {:+})", y - x),
            _ => String::new(),
        };
        let a_fmt = av.map_or("-".to_string(), |v| v.to_string());
        let b_fmt = bv.map_or("-".to_string(), |v| v.to_string());
        out.push_str(&format!("  {k}: {a_fmt} vs {b_fmt}{delta}\n"));
    }
    out.truncate(out.trim_end().len());
    out
}
