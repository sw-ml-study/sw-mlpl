//! Saga 32 step 004: `compare(name_a, name_b)` builtin dispatch
//! and record-comparison helpers extracted from `experiment.rs`
//! to keep the orchestrator under the sw-checklist function-count budget.

use mlpl_parser::Expr;

use crate::experiment::{ExperimentRecord, read_records_from_disk};

/// `compare(name_a, name_b)` builtin dispatch. Returns a
/// `Value::Str` containing a side-by-side of the most recent
/// runs with each name. Errors if either name has no records
/// or args are malformed.
pub(crate) fn dispatch_compare(
    args: &[Expr],
    env: &mut crate::env::Environment,
) -> Result<mlpl_eval_types::Value, mlpl_eval_types::EvalError> {
    if args.len() != 2 {
        return Err(mlpl_eval_types::EvalError::BadArity {
            func: "compare".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let Expr::StrLit(a, _) = &args[0] else {
        return Err(mlpl_eval_types::EvalError::Unsupported(
            "compare: arguments must be string literals".into(),
        ));
    };
    let Expr::StrLit(b, _) = &args[1] else {
        return Err(mlpl_eval_types::EvalError::Unsupported(
            "compare: arguments must be string literals".into(),
        ));
    };
    let ra = latest_by_name(env, a).ok_or_else(|| {
        mlpl_eval_types::EvalError::Unsupported(format!("compare: no run named {a:?}"))
    })?;
    let rb = latest_by_name(env, b).ok_or_else(|| {
        mlpl_eval_types::EvalError::Unsupported(format!("compare: no run named {b:?}"))
    })?;
    Ok(mlpl_eval_types::Value::Str(render_compare(&ra, &rb)))
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
