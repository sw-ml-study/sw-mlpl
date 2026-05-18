//! One-shot tool: write the tiny checked-in PNG / JPEG
//! decode fixtures under
//! `crates/mlpl-eval/tests/data/`. Saga 29 step 003.
//!
//! Run once to (re)generate the fixtures, then commit the
//! resulting files alongside the test that consumes them.
//! `cargo run --example build_tiny_fixtures
//! --features image-io -p mlpl-eval`.

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "build_tiny_fixtures requires the `image-io` feature; rebuild \
         with --features image-io"
    );
    std::process::exit(1);
}

#[cfg(feature = "image-io")]
use std::fs;
#[cfg(feature = "image-io")]
use std::io::BufWriter;
#[cfg(feature = "image-io")]
use std::path::PathBuf;
#[cfg(feature = "image-io")]
use std::process;

#[cfg(feature = "image-io")]
fn main() {
    if let Err(e) = run() {
        eprintln!("build_tiny_fixtures: {e}");
        process::exit(1);
    }
}

#[cfg(feature = "image-io")]
fn run() -> Result<(), String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| "CARGO_MANIFEST_DIR not set".to_string())?;
    let out_dir = PathBuf::from(&manifest_dir).join("tests/data");
    fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir: {e}"))?;

    // 4x4 PNG: gradient red top-left to blue bottom-right.
    // Pixel layout: [y, x] row-major, RGB triples.
    let mut rgb = Vec::with_capacity(4 * 4 * 3);
    for y in 0..4u8 {
        for x in 0..4u8 {
            // Red decreases left-to-right, blue increases.
            let r = 255 - (x * 85);
            let g = (x + y) * 32;
            let b = y * 85;
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    let png_path = out_dir.join("tiny_4x4.png");
    let file = fs::File::create(&png_path).map_err(|e| format!("create png: {e}"))?;
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, 4, 4);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("png header: {e}"))?;
    writer
        .write_image_data(&rgb)
        .map_err(|e| format!("png data: {e}"))?;
    println!("wrote {} ({} pixels)", png_path.display(), rgb.len() / 3);
    Ok(())
}
