//! Experiment record persistence + snapshotting, split out of
//! `experiment` so each module stays small: collect the metric / param
//! snapshots off the environment, and read/write run records under the
//! experiment dir. The evaluator and the `:experiments` registry render
//! stay in `experiment`.

use std::collections::BTreeMap;
use std::path::Path;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_state::{ExperimentRecord, ParamShape};

pub(crate) fn collect_metrics(env: &Environment) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for (name, arr) in env.vars_iter() {
        if name.ends_with("_metric") && arr.rank() == 0 {
            out.insert(name.clone(), arr.data()[0]);
        }
    }
    out
}

pub(crate) fn collect_param_shapes(env: &Environment) -> BTreeMap<String, ParamShape> {
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

pub(crate) fn write_record_to_disk(dir: &Path, rec: &ExperimentRecord) -> Result<(), String> {
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
