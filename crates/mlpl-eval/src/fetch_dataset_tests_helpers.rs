//! Shared helpers for the fetch_dataset test sibling files.
//! pub(super) so all sibling test modules (declared in
//! `fetch_dataset.rs`) can import them.

use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

pub(super) fn temp_dir(suffix: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("mlpl-fetch-dataset-{suffix}"));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

pub(super) fn write_tiny_png(path: &Path, color: [u8; 3]) {
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
