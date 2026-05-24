//! Compare-and-report helpers for `check_or_print_golden`.
//! Sample a pixel as hex, write the diagnostic PNG, format the
//! print-golden output, and produce a diff string when the
//! actual sampled hexes disagree with the golden array (within
//! per-channel tolerance).

use std::path::{Path, PathBuf};

use tiny_skia::Pixmap;

const TOL_PER_CHANNEL: u8 = 3;
/// Half-side of the median-sampling region. A `HALF=2` setting
/// samples the 5x5 pixel square centered on the test's nominal
/// `(x, y)` -- 25 pixels per logical sample -- and picks the
/// per-channel median. That swallows anti-aliased dot edges and
/// 1-pixel raster jitter without losing the fidelity we'd lose
/// from naive averaging across a color boundary.
const HALF: i32 = 2;

pub(crate) fn sample_hex(raster: &Pixmap, x: u32, y: u32) -> String {
    let (w, h) = (raster.width() as i32, raster.height() as i32);
    assert!(
        (x as i32) < w && (y as i32) < h,
        "sample_hex: ({x}, {y}) out of {w}x{h} canvas"
    );
    let mut rs = Vec::with_capacity(25);
    let mut gs = Vec::with_capacity(25);
    let mut bs = Vec::with_capacity(25);
    for dy in -HALF..=HALF {
        for dx in -HALF..=HALF {
            let nx = (x as i32 + dx).clamp(0, w - 1) as u32;
            let ny = (y as i32 + dy).clamp(0, h - 1) as u32;
            if let Some(p) = raster.pixel(nx, ny) {
                rs.push(p.red());
                gs.push(p.green());
                bs.push(p.blue());
            }
        }
    }
    rs.sort_unstable();
    gs.sort_unstable();
    bs.sort_unstable();
    let mid = rs.len() / 2;
    format!("#{:02x}{:02x}{:02x}", rs[mid], gs[mid], bs[mid])
}

pub(crate) fn write_fail_png(name: &str, raster: &Pixmap) -> PathBuf {
    let dir = PathBuf::from("/tmp/mlpl-reg-fail");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{name}.png"));
    match raster.encode_png() {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("mlpl-reg: failed to write {path:?}: {e}");
            }
        }
        Err(e) => eprintln!("mlpl-reg: failed to encode PNG: {e:?}"),
    }
    path
}

pub(crate) fn print_golden(name: &str, actual: &[String], png_path: &Path) {
    eprintln!("\nMLPL_REG_PRINT_GOLDEN=1 -- sampled hexes for `{name}`:");
    eprintln!("const GOLDEN: &[&str] = &[");
    for hex in actual {
        eprintln!("    {hex:?},");
    }
    eprintln!("];");
    eprintln!("(inspect the PNG at {png_path:?} before pasting)\n");
}

pub(crate) fn diff(samples: &[(u32, u32)], actual: &[String], golden: &[&str]) -> Option<String> {
    if actual.len() != golden.len() {
        return Some(format!(
            "  sample count mismatch: SAMPLE_POINTS={} GOLDEN={}\n",
            samples.len(),
            golden.len()
        ));
    }
    let mut diffs = String::new();
    for (i, ((x, y), (a, g))) in samples
        .iter()
        .zip(actual.iter().zip(golden.iter()))
        .enumerate()
    {
        if !within_tolerance(a, g) {
            diffs.push_str(&format!(
                "  sample {i} at ({x}, {y}): expected {g}, got {a}\n"
            ));
        }
    }
    if diffs.is_empty() { None } else { Some(diffs) }
}

fn within_tolerance(a: &str, b: &str) -> bool {
    if a.len() != 7 || b.len() != 7 || !a.starts_with('#') || !b.starts_with('#') {
        return a == b;
    }
    [1usize, 3, 5].iter().all(|&i| {
        let av = u8::from_str_radix(&a[i..i + 2], 16).unwrap_or(0);
        let bv = u8::from_str_radix(&b[i..i + 2], 16).unwrap_or(0);
        (i16::from(av) - i16::from(bv)).abs() <= i16::from(TOL_PER_CHANNEL)
    })
}
