//! Saga 32 step 006: IO + decode helpers extracted from
//! `fetch_dataset.rs` to retire its function-count FAIL.
//! Pure side-effecting helpers: download, hash verify, tar
//! extract, image-directory decode, and DenseArray packing.

#![cfg(feature = "image-io")]

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use mlpl_array::{DenseArray, Shape};
use sha2::Digest;

use crate::error::EvalError;
use crate::value::Value;

use crate::fetch_dataset::{io_err, label_for};

pub(crate) fn ensure_tarball(
    path: &Path,
    url: &str,
    expected_sha256: &str,
) -> Result<(), EvalError> {
    if path.exists() {
        let got = sha256_of(path)?;
        if got == expected_sha256 {
            return Ok(());
        }
        return Err(EvalError::Unsupported(format!(
            "fetch_dataset: {} sha256 mismatch (expected {expected_sha256}, \
             got {got}). Delete the file and retry.",
            path.display()
        )));
    }
    let resp = ureq::get(url)
        .call()
        .map_err(|e| EvalError::Unsupported(format!("fetch_dataset: HTTP GET {url}: {e}")))?;
    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| EvalError::Unsupported(format!("fetch_dataset: read body from {url}: {e}")))?;
    let got = format!("{:x}", sha2::Sha256::digest(&buf));
    if got != expected_sha256 {
        return Err(EvalError::Unsupported(format!(
            "fetch_dataset: downloaded {url} but sha256 was {got} \
             (expected {expected_sha256})"
        )));
    }
    let mut f = fs::File::create(path).map_err(|e| io_err(path, e))?;
    f.write_all(&buf).map_err(|e| io_err(path, e))?;
    Ok(())
}

pub(crate) fn sha256_of(path: &Path) -> Result<String, EvalError> {
    let mut f = fs::File::open(path).map_err(|e| io_err(path, e))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| io_err(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn extract_tarball(tar_path: &Path, dest: &Path) -> Result<(), EvalError> {
    let f = fs::File::open(tar_path).map_err(|e| io_err(tar_path, e))?;
    let gz = flate2::read::GzDecoder::new(f);
    let mut archive = tar::Archive::new(gz);
    archive.unpack(dest).map_err(|e| {
        EvalError::Unsupported(format!(
            "fetch_dataset: tar extract {} -> {}: {e}",
            tar_path.display(),
            dest.display()
        ))
    })?;
    Ok(())
}

/// Scan a directory for PNG / JPEG files, decode + resize +
/// normalize each, build the `Record{X, Y, names}` payload.
/// Cat vs dog is encoded in the filename per Oxford-IIIT Pet
/// convention: capitalized prefix = cat (label 0), lowercase
/// prefix = dog (label 1).
pub(crate) fn decode_directory_to_record(
    images_dir: &Path,
    h: usize,
    w: usize,
) -> Result<Value, EvalError> {
    let mut paths: Vec<PathBuf> = fs::read_dir(images_dir)
        .map_err(|e| io_err(images_dir, e))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| is_image_ext(p))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(EvalError::Unsupported(format!(
            "fetch_dataset: no PNG or JPEG files in {}",
            images_dir.display()
        )));
    }
    let mut x_data: Vec<f64> = Vec::with_capacity(paths.len() * 3 * h * w);
    let mut y_data: Vec<f64> = Vec::with_capacity(paths.len());
    let mut names: Vec<String> = Vec::with_capacity(paths.len());
    for path in &paths {
        x_data.extend_from_slice(&mlpl_eval_image::decode_and_resize(path, h, w)?);
        let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
            EvalError::Unsupported(format!(
                "fetch_dataset: non-utf8 filename {}",
                path.display()
            ))
        })?;
        y_data.push(label_for(name) as f64);
        names.push(name.to_string());
    }
    build_record(paths.len(), 3, h, w, x_data, y_data, names)
}

pub(crate) fn is_image_ext(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
        .unwrap_or(false)
}

pub(crate) fn build_record(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    names: Vec<String>,
) -> Result<Value, EvalError> {
    let x_arr = DenseArray::new(Shape::new(vec![n, c, h, w]), x_data)?.with_labels(vec![
        Some("batch".to_string()),
        Some("channel".to_string()),
        Some("y".to_string()),
        Some("x".to_string()),
    ])?;
    let y_arr = DenseArray::new(Shape::new(vec![n]), y_data)?
        .with_labels(vec![Some("batch".to_string())])?;
    let mut fields = BTreeMap::new();
    fields.insert("X".to_string(), Value::Array(x_arr));
    fields.insert("Y".to_string(), Value::Array(y_arr));
    fields.insert("names".to_string(), Value::StrList { items: names });
    Ok(Value::Record { fields })
}
