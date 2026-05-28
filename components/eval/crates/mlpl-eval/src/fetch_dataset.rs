//! `fetch_dataset(name)` -- download + verify + extract +
//! decode the Oxford-IIIT Pet dataset at the demo's 128x128
//! resolution. Saga 29 step 004.
//!
//! Native-only via the `image-io` Cargo feature; the WASM
//! REPL gets a clean error pointing at
//! `load_preloaded("pets_tiny")` instead. The pipeline
//! caches by file existence -- if the tarball is already
//! sitting in `$MLPL_DATA_DIR/oxford-iiit-pet/images.tar.gz`
//! (sha256-verified), we skip the download; if
//! `images/` is already extracted, we skip the extract. So
//! repeat calls within a session are cheap, and a
//! pre-populated `data/oxford-iiit-pet/` checkout (the
//! gitignored layout that step 003 set up) bypasses HTTP
//! entirely.
//!
//! Dataset resolution: each name maps to a `DatasetSpec`
//! that records the canonical upstream URL, the sha256 of
//! the tarball we know how to handle, the on-disk layout,
//! and the demo's target resolution. Adding a new dataset
//! is a single-line entry plus a test fixture; we deliberately
//! keep the registry tiny.

#![cfg(feature = "image-io")]

use std::fs;
use std::path::{Path, PathBuf};

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

const OXFORD_PET_URL: &str = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz";
const OXFORD_PET_SHA256: &str = "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d";
const OXFORD_PET_H: usize = 128;
const OXFORD_PET_W: usize = 128;

struct DatasetSpec {
    url: &'static str,
    sha256: &'static str,
    /// Relative subdir under `$MLPL_DATA_DIR` for everything
    /// related to this dataset.
    subdir: &'static str,
    /// Tarball filename inside the subdir.
    tarball: &'static str,
    /// Image-bearing subdirectory inside the tarball (after
    /// extract).
    images_subdir: &'static str,
    target_h: usize,
    target_w: usize,
}

fn lookup(name: &str) -> Result<DatasetSpec, EvalError> {
    match name {
        "oxford_iiit_pet" => Ok(DatasetSpec {
            url: OXFORD_PET_URL,
            sha256: OXFORD_PET_SHA256,
            subdir: "oxford-iiit-pet",
            tarball: "images.tar.gz",
            images_subdir: "images",
            target_h: OXFORD_PET_H,
            target_w: OXFORD_PET_W,
        }),
        _ => Err(EvalError::Unsupported(format!(
            "fetch_dataset(\"{name}\"): unknown dataset (the v0.21 \
             registry only ships \"oxford_iiit_pet\"; PRs welcome)"
        ))),
    }
}

/// Entry point called from `eval::eval_expr`.
pub(crate) fn eval(env: &Environment, name: &str) -> Result<Value, EvalError> {
    let spec = lookup(name)?;
    let root = resolve_data_root(env, name)?;
    let dataset_root = root.join(spec.subdir);
    fs::create_dir_all(&dataset_root).map_err(|e| io_err(&dataset_root, e))?;
    let tarball_path = dataset_root.join(spec.tarball);
    let images_path = dataset_root.join(spec.images_subdir);
    crate::fetch_io::ensure_tarball(&tarball_path, spec.url, spec.sha256)?;
    if !images_path.exists() {
        crate::fetch_io::extract_tarball(&tarball_path, &dataset_root)?;
    }
    crate::fetch_io::decode_directory_to_record(&images_path, spec.target_h, spec.target_w)
}

/// Resolve where dataset files live. Priority: `$MLPL_DATA_DIR`
/// environment variable; then the `Environment::data_dir`
/// sandbox set via `--data-dir`. Without either, raise a
/// clean tutoring error -- a 792 MB download should never be
/// implicit.
fn resolve_data_root(env: &Environment, name: &str) -> Result<PathBuf, EvalError> {
    if let Ok(dir) = std::env::var("MLPL_DATA_DIR")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    if let Some(p) = env.data_dir() {
        return Ok(p.clone());
    }
    Err(EvalError::Unsupported(format!(
        "fetch_dataset(\"{name}\"): no data directory configured. \
         Set the MLPL_DATA_DIR env var (where the tarball will be \
         cached) or start the terminal REPL with --data-dir <path>."
    )))
}

/// Download the tarball to `path` if it isn't already there.
/// Always verifies sha256 -- if a stale file is sitting at
/// the path with the wrong hash, we error rather than silently
/// trust it.
/// Cat (uppercase prefix) -> 0, dog (lowercase prefix) -> 1.
/// Non-alphabetic prefix surfaces as 255 so the caller can
/// spot junk in the dataset.
pub(crate) fn label_for(name: &str) -> u8 {
    match name.chars().next() {
        Some(c) if c.is_ascii_uppercase() => 0,
        Some(c) if c.is_ascii_lowercase() => 1,
        _ => 255,
    }
}

pub(crate) fn io_err(path: &Path, e: std::io::Error) -> EvalError {
    EvalError::Unsupported(format!("fetch_dataset: {}: {e}", path.display()))
}

// Saga 33 step 022: tests split across 4 sibling files so
// each one stays under the 7-fn Module-Function-Count FAIL
// line. The "_test_helpers" submodule is shared by the other
// three via super::_test_helpers::* imports.
#[cfg(test)]
#[path = "fetch_dataset_tests_helpers.rs"]
mod _test_helpers;
#[cfg(test)]
#[path = "fetch_dataset_tests_archive.rs"]
mod _tests_archive;
#[cfg(test)]
#[path = "fetch_dataset_tests_basic.rs"]
mod _tests_basic;
#[cfg(test)]
#[path = "fetch_dataset_tests_lookup.rs"]
mod _tests_lookup;
