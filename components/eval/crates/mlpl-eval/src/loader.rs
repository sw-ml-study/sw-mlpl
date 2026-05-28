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
//!
//! Saga 33 step 017: path sandboxing + CSV parsing extracted to
//! `mlpl-loader-helpers` (pure-data helpers, no env / Value
//! dependency).

#[cfg(feature = "image-io")]
use std::path::PathBuf;

#[cfg(feature = "image-io")]
use mlpl_array::{DenseArray, Shape};
use mlpl_loader_helpers::{parse_csv, resolve_in_sandbox};

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

/// Dispatch `load(path)`.
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
        parse_csv(&contents, path)
            .map(Value::Array)
            .map_err(EvalError::from)
    } else {
        Ok(Value::Str(contents))
    }
}

/// Dispatch `load_preloaded(name)`.
pub(crate) fn eval_load_preloaded(name: &str) -> Result<Value, EvalError> {
    if name == "pets_tiny" {
        return crate::pets_tiny::load();
    }
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

/// Dispatch `load_images(dir, [H, W])`. Native-only via the
/// `image-io` feature; the WASM-side stub raises a clean
/// error pointing users at `load_preloaded("pets_tiny")`.
#[cfg(feature = "image-io")]
pub(crate) fn eval_load_images(
    env: &Environment,
    dir: &str,
    h: usize,
    w: usize,
) -> Result<Value, EvalError> {
    let Some(root) = env.data_dir() else {
        return Err(EvalError::Unsupported(format!(
            "load_images(\"{dir}\"): filesystem access disabled (try \
             load_preloaded(\"pets_tiny\") in the web REPL, or start the \
             terminal REPL with --data-dir <path>)"
        )));
    };
    let resolved = resolve_in_sandbox(root, dir)?;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&resolved)
        .map_err(|e| EvalError::Unsupported(format!("load_images(\"{dir}\"): {e}")))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    let e = e.to_ascii_lowercase();
                    e == "png" || e == "jpg" || e == "jpeg"
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(EvalError::Unsupported(format!(
            "load_images(\"{dir}\"): no PNG or JPEG files in {}",
            resolved.display()
        )));
    }
    let per_image = 3 * h * w;
    let mut data = Vec::with_capacity(paths.len() * per_image);
    for path in &paths {
        let pixels = crate::image_io::decode_and_resize(path, h, w)?;
        data.extend_from_slice(&pixels);
    }
    let labels = vec![
        Some("batch".to_string()),
        Some("channel".to_string()),
        Some("y".to_string()),
        Some("x".to_string()),
    ];
    let arr = DenseArray::new(Shape::new(vec![paths.len(), 3, h, w]), data)?.with_labels(labels)?;
    Ok(Value::Array(arr))
}

/// WASM stub: `load_images` is native-only because the
/// decoders aren't available in the WASM target. Tell the
/// user where to go instead.
#[cfg(not(feature = "image-io"))]
pub(crate) fn eval_load_images(
    _env: &Environment,
    dir: &str,
    _h: usize,
    _w: usize,
) -> Result<Value, EvalError> {
    Err(EvalError::Unsupported(format!(
        "load_images(\"{dir}\"): PNG / JPEG decode is disabled in this \
         build (the WASM REPL ships the pre-decoded `pets_tiny` fixture \
         -- use `load_preloaded(\"pets_tiny\")` instead). Rebuild a \
         native binary with `--features mlpl-eval/image-io` to enable \
         live decode."
    )))
}

/// Dispatch `fetch_dataset(name)`. Native-only via
/// `image-io`; WASM stub points users at the preloaded
/// fixture.
#[cfg(feature = "image-io")]
pub(crate) fn eval_fetch_dataset(env: &Environment, name: &str) -> Result<Value, EvalError> {
    crate::fetch_dataset::eval(env, name)
}

#[cfg(not(feature = "image-io"))]
pub(crate) fn eval_fetch_dataset(_env: &Environment, name: &str) -> Result<Value, EvalError> {
    Err(EvalError::Unsupported(format!(
        "fetch_dataset(\"{name}\"): live dataset download is disabled \
         in this build (the WASM REPL ships the pre-decoded \
         `pets_tiny` fixture -- use `load_preloaded(\"pets_tiny\")` \
         instead). Rebuild a native binary with \
         `--features mlpl-eval/image-io` to enable fetching."
    )))
}
