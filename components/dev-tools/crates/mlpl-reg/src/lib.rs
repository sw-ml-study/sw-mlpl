//! Visual regression test harness for MLPL demos.
//!
//! Saga 33 step 026: rasterize a demo's SVG output in memory
//! (via resvg + tiny-skia, no PNG files ever committed),
//! sample a small fixed table of pixel coordinates, and
//! compare the sampled hex colors against an inline golden
//! array. Designed for catching renderer drift end-to-end:
//! the demo is run through the same eval pipeline the web
//! playground uses, so any change to lexing, parsing,
//! training, or rendering surfaces as a hex-color mismatch.
//!
//! Pixmap canvas size is fixed at 400x300 to match MLPL's
//! intrinsic viewBox. Rendering at a larger raster would
//! re-scale the (x, y) sample tables and silently invalidate
//! existing goldens.
//!
//! BOOTSTRAP workflow for capturing a new golden:
//!   1. Write a new test file under `crates/mlpl-reg/tests/`
//!      with `SAMPLE_POINTS` populated and `GOLDEN` empty.
//!   2. Run `MLPL_REG_PRINT_GOLDEN=1 cargo test -p mlpl-reg
//!      --test <name> --release -- --nocapture`. The harness
//!      writes the actual PNG to `/tmp/mlpl-reg-fail/<name>.png`
//!      and prints the sampled hexes as a copy-pasteable array.
//!   3. OPEN THE PNG and visually verify the rendered output
//!      is what you expect for the demo. Do not skip this --
//!      a golden captured silently from a buggy system silently
//!      calcifies the bug (see `docs/moons-post-mortem.md`).
//!   4. Paste the printed array into the test file as the
//!      `GOLDEN` const.
//!   5. Re-run without the env flag; test should now pass.

mod compare;

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};
use tiny_skia::Pixmap;

const CANVAS_W: u32 = 400;
const CANVAS_H: u32 = 300;
const PRINT_GOLDEN_ENV: &str = "MLPL_REG_PRINT_GOLDEN";

/// Run an MLPL source program end-to-end through the same
/// eval pipeline the web playground uses, and return the
/// final expression's string value. Demos must end on a viz
/// builtin (`boundary_2d(...)`, `loss_curve(...)`, etc.) so
/// the return is the SVG text.
pub fn run_demo_to_svg(mlpl_source: &str) -> Result<String, String> {
    let tokens = lex(mlpl_source).map_err(|e| format!("lex: {e:?}"))?;
    let stmts = parse(&tokens).map_err(|e| format!("parse: {e:?}"))?;
    let mut env = Environment::default();
    match eval_program_value(&stmts, &mut env).map_err(|e| format!("eval: {e}"))? {
        Value::Str(s) => Ok(s),
        other => Err(format!(
            "demo must end on a viz builtin returning Value::Str, got {other:?}"
        )),
    }
}

/// Rasterize an SVG string to an in-memory 400x300 pixel
/// buffer. The canvas size is fixed to match MLPL's intrinsic
/// viewBox so the inline (x, y) sample coordinates in each
/// test file map to the same pixels they were captured from.
pub fn rasterize(svg: &str) -> Result<Pixmap, String> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default())
        .map_err(|e| format!("usvg parse: {e:?}"))?;
    let mut pixmap =
        Pixmap::new(CANVAS_W, CANVAS_H).ok_or("tiny_skia: failed to allocate pixmap")?;
    let tx = tiny_skia::Transform::identity();
    resvg::render(&tree, tx, &mut pixmap.as_mut());
    Ok(pixmap)
}

/// Compare sampled hex colors against a golden array, OR --
/// when `MLPL_REG_PRINT_GOLDEN=1` -- print the sampled hexes
/// as a copy-pasteable Rust array (and write a diagnostic PNG
/// to `/tmp/mlpl-reg-fail/<name>.png` for the human to
/// inspect before pasting into the golden). On assertion
/// failure the harness writes the actual PNG to the same path
/// so the human can see what changed.
///
/// Each "sample" is the per-channel median over the 5x5 pixel
/// square centered on its nominal (x, y) coordinate (25
/// pixels per sample). Comparison uses a per-channel RGB
/// tolerance of +/-3. Both layers (region median + channel
/// tolerance) defend against anti-aliasing jitter and
/// near-misses on scatter-dot edges.
pub fn check_or_print_golden(name: &str, raster: &Pixmap, samples: &[(u32, u32)], golden: &[&str]) {
    let actual: Vec<String> = samples
        .iter()
        .map(|(x, y)| compare::sample_hex(raster, *x, *y))
        .collect();
    if std::env::var(PRINT_GOLDEN_ENV).is_ok() {
        let png_path = compare::write_fail_png(name, raster);
        compare::print_golden(name, &actual, &png_path);
        return;
    }
    if let Some(diffs) = compare::diff(samples, &actual, golden) {
        let png_path = compare::write_fail_png(name, raster);
        panic!(
            "\nvisual regression in `{name}`:\n{diffs}\nactual PNG written to {png_path:?}\nrerun with {PRINT_GOLDEN_ENV}=1 to refresh after visual inspection.\n"
        );
    }
}
