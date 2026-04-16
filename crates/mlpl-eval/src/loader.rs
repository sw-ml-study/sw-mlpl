//! Filesystem and preloaded-corpus loaders (Saga 12 step 001).
//!
//! Two builtins live here:
//!
//! - `load("relative-path")` reads a file under the
//!   `Environment::data_dir` sandbox. `.csv` files parse as numeric
//!   matrices (stripping a non-numeric header row if present);
//!   `.txt` files return their whole contents as a `Value::Str`.
//!   Absolute paths and `..`-traversal outside the sandbox error
//!   cleanly. When `data_dir` is `None` (the web REPL), `load`
//!   errors with a pointer to `load_preloaded`.
//! - `load_preloaded("name")` looks up a small in-memory corpus
//!   registry so the web REPL has a fs-free path for the
//!   Tokenizing-Text tutorial lesson in step 009.

use std::path::{Component, Path, PathBuf};

use mlpl_array::{DenseArray, Shape};

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;

/// Compiled-in preloaded corpora. Keeping the registry tiny and
/// explicit; the web REPL's WASM binary carries these as string
/// literals.
const PRELOADED: &[(&str, &str)] = &[
    (
        "tiny_corpus",
        "the quick brown fox jumps over the lazy dog.\n\
         pack my box with five dozen liquor jugs.\n",
    ),
    (
        "tiny_shakespeare_snippet",
        include_str!("../data/tiny_shakespeare_snippet.txt"),
    ),
];

/// Dispatch `load(path)`. Called from `eval::eval_expr` when the
/// evaluator sees a `FnCall { name == "load", args.len() == 1 }`
/// whose single arg evaluates to a `Value::Str`.
pub(crate) fn eval_load(env: &Environment, path: &str) -> Result<Value, EvalError> {
    let Some(root) = env.data_dir() else {
        return Err(EvalError::Unsupported(format!(
            "load(\"{path}\"): filesystem access disabled (try load_preloaded(\"...\") \
             in the web REPL, or start the terminal REPL with --data-dir <path>)"
        )));
    };
    let resolved = resolve_in_sandbox(root, path)?;
    let contents = std::fs::read_to_string(&resolved)
        .map_err(|e| EvalError::Unsupported(format!("load(\"{path}\"): {e}")))?;
    if path.ends_with(".csv") {
        parse_csv(&contents, path).map(Value::Array)
    } else {
        Ok(Value::Str(contents))
    }
}

/// Dispatch `load_preloaded(name)`.
pub(crate) fn eval_load_preloaded(name: &str) -> Result<Value, EvalError> {
    PRELOADED
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, body)| Value::Str((*body).to_string()))
        .ok_or_else(|| {
            EvalError::Unsupported(format!(
                "load_preloaded(\"{name}\"): unknown preloaded corpus"
            ))
        })
}

/// Resolve a caller-supplied relative path under a sandbox root.
/// Absolute paths and any component that escapes the root via `..`
/// are rejected.
fn resolve_in_sandbox(root: &Path, relative: &str) -> Result<PathBuf, EvalError> {
    let rel = Path::new(relative);
    if rel.is_absolute() {
        return Err(EvalError::Unsupported(format!(
            "load(\"{relative}\"): absolute paths are rejected; paths are relative \
             to the sandbox root {}",
            root.display()
        )));
    }
    // Walk components manually so we catch `..` escapes without
    // touching the filesystem (canonicalize would dereference
    // symlinks, which we don't want).
    let mut depth: i64 = 0;
    for comp in rel.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {
                depth += i64::from(matches!(comp, Component::Normal(_)))
            }
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(EvalError::Unsupported(format!(
                        "load(\"{relative}\"): path escapes sandbox root {}",
                        root.display()
                    )));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(EvalError::Unsupported(format!(
                    "load(\"{relative}\"): rooted components not permitted inside \
                     sandbox {}",
                    root.display()
                )));
            }
        }
    }
    Ok(root.join(rel))
}

/// Parse a CSV string into a 2-D `DenseArray`. Handles comma
/// delimiters only. If the first row contains any non-numeric
/// token it is treated as a header and skipped. Ragged rows or
/// non-numeric data rows surface as `EvalError::Unsupported`.
fn parse_csv(text: &str, path: &str) -> Result<DenseArray, EvalError> {
    let mut rows: Vec<Vec<String>> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.split(',').map(|c| c.trim().to_string()).collect())
        .collect();
    if rows.is_empty() {
        return Err(EvalError::Unsupported(format!(
            "load(\"{path}\"): file contains no data rows"
        )));
    }
    // Header detection: if any first-row cell fails to parse as f64,
    // treat the first row as a header and drop it.
    let first_is_header = rows[0].iter().any(|cell| cell.parse::<f64>().is_err());
    if first_is_header {
        rows.remove(0);
        if rows.is_empty() {
            return Err(EvalError::Unsupported(format!(
                "load(\"{path}\"): header-only file with no data rows"
            )));
        }
    }
    let cols = rows[0].len();
    let mut data = Vec::with_capacity(rows.len() * cols);
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != cols {
            return Err(EvalError::Unsupported(format!(
                "load(\"{path}\"): ragged rows (row {row_idx} has {} cols, \
                 expected {cols})",
                row.len()
            )));
        }
        for cell in row {
            let v: f64 = cell.parse().map_err(|_| {
                EvalError::Unsupported(format!(
                    "load(\"{path}\"): non-numeric cell \"{cell}\" at row {row_idx}"
                ))
            })?;
            data.push(v);
        }
    }
    DenseArray::new(Shape::new(vec![rows.len(), cols]), data).map_err(EvalError::ArrayError)
}
