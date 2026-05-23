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
use crate::error::EvalError;
use crate::value::Value;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufWriter;

    fn temp_dir(suffix: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("mlpl-fetch-dataset-{suffix}"));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_tiny_png(path: &Path, color: [u8; 3]) {
        // 4x4 solid color PNG.
        let mut rgb = Vec::with_capacity(4 * 4 * 3);
        for _ in 0..16 {
            rgb.extend_from_slice(&color);
        }
        let f = fs::File::create(path).unwrap();
        let mut enc = png::Encoder::new(BufWriter::new(f), 4, 4);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().unwrap();
        w.write_image_data(&rgb).unwrap();
    }

    #[test]
    fn label_for_assigns_cat_zero_dog_one() {
        assert_eq!(label_for("Abyssinian_1.jpg"), 0);
        assert_eq!(label_for("beagle_3.jpg"), 1);
        assert_eq!(label_for("12345.jpg"), 255);
    }

    #[test]
    fn sha256_of_matches_known_content() {
        let tmp = temp_dir("sha256");
        let p = tmp.join("hello.txt");
        fs::write(&p, b"hello").unwrap();
        // SHA-256("hello") is well-known.
        assert_eq!(
            crate::fetch_io::sha256_of(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn ensure_tarball_passes_when_existing_hash_matches() {
        let tmp = temp_dir("ensure-ok");
        let p = tmp.join("blob");
        fs::write(&p, b"abc").unwrap();
        // SHA-256("abc"):
        let want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        // Won't hit the network since the file exists + hash matches.
        crate::fetch_io::ensure_tarball(&p, "http://invalid.example/never", want).unwrap();
    }

    #[test]
    fn ensure_tarball_errors_on_stale_hash() {
        let tmp = temp_dir("ensure-stale");
        let p = tmp.join("blob");
        fs::write(&p, b"abc").unwrap();
        let err = crate::fetch_io::ensure_tarball(&p, "http://invalid.example/never", "deadbeef")
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("sha256 mismatch"), "got: {msg}");
    }

    #[test]
    fn extract_tarball_unpacks_files_into_dest() {
        let tmp = temp_dir("extract");
        // Build a minimal tar.gz: one file "hello.txt" with
        // body "hi".
        let tar_path = tmp.join("test.tar.gz");
        let body = b"hi";
        let f = fs::File::create(&tar_path).unwrap();
        let gz = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        let mut archive = tar::Builder::new(gz);
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "hello.txt", &body[..])
            .unwrap();
        archive.into_inner().unwrap().finish().unwrap();
        let dest = tmp.join("out");
        fs::create_dir_all(&dest).unwrap();
        crate::fetch_io::extract_tarball(&tar_path, &dest).unwrap();
        let got = fs::read_to_string(dest.join("hello.txt")).unwrap();
        assert_eq!(got, "hi");
    }

    #[test]
    fn decode_directory_to_record_returns_expected_record() {
        let tmp = temp_dir("decode-dir");
        // 2 "cats" (uppercase prefix) + 2 "dogs" (lowercase).
        write_tiny_png(&tmp.join("Abyssinian_1.png"), [255, 0, 0]);
        write_tiny_png(&tmp.join("Bombay_1.png"), [0, 255, 0]);
        write_tiny_png(&tmp.join("beagle_1.png"), [0, 0, 255]);
        write_tiny_png(&tmp.join("pug_1.png"), [128, 128, 128]);
        let v = crate::fetch_io::decode_directory_to_record(&tmp, 4, 4).unwrap();
        let Value::Record { fields } = v else {
            panic!("expected Record");
        };
        let x = match fields.get("X") {
            Some(Value::Array(a)) => a,
            _ => panic!("X missing"),
        };
        assert_eq!(x.shape().dims(), &[4, 3, 4, 4]);
        let labels = x.labels().expect("X must carry axis labels");
        let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
        assert_eq!(names, vec!["batch", "channel", "y", "x"]);
        let y = match fields.get("Y") {
            Some(Value::Array(a)) => a,
            _ => panic!("Y missing"),
        };
        // Sorted filenames: Abyssinian, Bombay, beagle, pug.
        // Sorting is ASCII (uppercase first), so cats land
        // at indices 0-1 and dogs at 2-3.
        assert_eq!(y.data(), &[0.0, 0.0, 1.0, 1.0]);
        let names_list = match fields.get("names") {
            Some(Value::StrList { items }) => items,
            _ => panic!("names missing"),
        };
        assert_eq!(names_list.len(), 4);
        assert!(names_list[0].starts_with("Abyssinian"));
        assert!(names_list[3].starts_with("pug"));
    }

    #[test]
    fn decode_directory_to_record_errors_on_empty_dir() {
        let tmp = temp_dir("decode-empty");
        let err = crate::fetch_io::decode_directory_to_record(&tmp, 4, 4).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no PNG or JPEG"), "got: {msg}");
    }

    #[test]
    fn lookup_known_dataset_returns_spec() {
        let s = lookup("oxford_iiit_pet").expect("known dataset");
        assert_eq!(s.subdir, "oxford-iiit-pet");
        assert_eq!(s.target_h, 128);
        assert_eq!(s.target_w, 128);
    }

    #[test]
    fn lookup_unknown_dataset_errors() {
        let res = lookup("nope");
        assert!(res.is_err(), "expected unknown dataset error");
        if let Err(e) = res {
            let msg = format!("{e}");
            assert!(msg.contains("unknown dataset"), "got: {msg}");
        }
    }
}
