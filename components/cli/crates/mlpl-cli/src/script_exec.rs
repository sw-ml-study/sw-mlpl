//! Host-side `run_script` implementation: evaluate a script file
//! in a FRESH Environment (definitions, test metadata, and
//! bindings cannot leak between files), through the chunked
//! include loader so `include` and file-accurate spans behave
//! exactly as in normal script runs. Outcomes come back as
//! STRUCTURED DATA: `ok({status, value, error, events_kind,
//! events})`; only infrastructure failures (unreadable /
//! unparsable source) are `err(...)`.

use std::collections::BTreeMap;
use std::path::Path;

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, RunScriptOpts, Value};

/// The hook `mlpl-repl` registers into every Environment.
pub fn run_script_value(path: &Path, opts: &RunScriptOpts) -> Value {
    let source_dir = opts
        .source_dir
        .clone()
        .or_else(|| path.parent().map(std::path::Path::to_path_buf));
    let loaded = match crate::include_script::load_script(path, source_dir.as_deref()) {
        Ok(l) => l,
        Err(msg) => return err_value(msg),
    };
    let mut env = Environment::new();
    env.fs_root = source_dir.clone();
    if let Some(dir) = &opts.data_dir {
        env.set_data_dir(dir.clone());
    }
    if opts.capture {
        env.test_event_lines = Some(Vec::new());
    }
    // The exit DI seam: a child calling exit(code) reports
    // structured status instead of killing the runner.
    env.exit_intercept = true;
    let mut last: Option<Value> = None;
    let mut error: Option<String> = None;
    for chunk in &loaded.chunks {
        env.current_source = Some(chunk.source.0.clone());
        env.set_pending_source(loaded.table.text(&chunk.source).map(String::from));
        let r = mlpl_eval::eval_program_value(&chunk.stmts, &mut env);
        env.set_pending_source(None);
        match r {
            Ok(v) => last = Some(v),
            Err(mlpl_eval::EvalError::ExitRequested(code)) => {
                error = Some(format!("__exit__{code}"));
                break;
            }
            Err(e) => {
                error = Some(format!("{e} (in {})", chunk.source.0));
                break;
            }
        }
    }
    ok_value(outcome_record(last, error, &mut env))
}

/// `{status, value, error, events_kind, events}` -- the child's
/// outcome as data. status: "ok" (final value Ok or plain),
/// "err" (final Result is Err), "error" (hard eval error).
fn outcome_record(
    last: Option<Value>,
    error: Option<String>,
    env: &mut Environment,
) -> BTreeMap<String, Value> {
    let (status, value, err_text) = match (&error, &last) {
        (Some(e), _) if e.starts_with("__exit__") => (
            "exit",
            e.trim_start_matches("__exit__").to_string(),
            String::new(),
        ),
        (Some(e), _) => ("error", String::new(), e.clone()),
        (None, Some(Value::Result { ok: false, payload })) => (
            "err",
            mlpl_value_structural::value_repr(payload),
            String::new(),
        ),
        (None, Some(v)) => ("ok", mlpl_value_structural::value_repr(v), String::new()),
        (None, None) => ("ok", String::new(), String::new()),
    };
    let events = env.test_event_lines.take().unwrap_or_default();
    BTreeMap::from([
        ("status".to_string(), Value::Str(status.to_string())),
        ("value".to_string(), Value::Str(value)),
        ("error".to_string(), Value::Str(err_text)),
        (
            "events_kind".to_string(),
            Value::Str("test_events".to_string()),
        ),
        ("events".to_string(), Value::StrList { items: events }),
    ])
}

fn ok_value(fields: BTreeMap<String, Value>) -> Value {
    Value::Result {
        ok: true,
        payload: Box::new(Value::Record { fields }),
    }
}

fn err_value(message: String) -> Value {
    Value::Result {
        ok: false,
        payload: Box::new(Value::Str(message)),
    }
}
