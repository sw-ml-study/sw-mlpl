//! Waffle / unit chart: a grid of unit blocks colored by outcome
//! category. Each input value is a category count, taken in order as
//! `[losses, ties, wins]` and rendered red / gray / green. A vector
//! draws one grid; an `R x K` matrix draws `R` stacked grids (e.g.
//! before/after). Used by the tic-tac-toe fine-tune demo to show
//! piles of lost games turn into wins.

use mlpl_array::DenseArray;
use mlpl_viz_core::{VizError, write_svg_close, write_svg_open_with_size};

const COLS: usize = 10;
const CELL: f64 = 16.0;
const GAP: f64 = 3.0;
const MARGIN: f64 = 12.0;
const BAND_GAP: f64 = 14.0;
// loss / tie / win -> catppuccin red / overlay gray / green.
const COLORS: [&str; 3] = ["#f38ba8", "#6c7086", "#a6e3a1"];

/// Render outcome counts as one or more waffle grids.
pub fn render_waffle(data: &DenseArray) -> Result<String, VizError> {
    let bands = bands_of(data)?;
    let width = MARGIN * 2.0 + COLS as f64 * (CELL + GAP) - GAP;
    let mut out = String::new();
    write_svg_open_with_size(&mut out, width, total_height(&bands));
    out.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#1e1e2e\"/>");
    let mut y = MARGIN;
    for band in &bands {
        y = render_band(&mut out, band, y);
    }
    write_svg_close(&mut out);
    Ok(out)
}

/// Split the array into per-band count vectors: a vector is one band,
/// an `R x K` matrix is `R` bands.
fn bands_of(data: &DenseArray) -> Result<Vec<Vec<f64>>, VizError> {
    let raw = data.data();
    match data.shape().dims() {
        [] | [_] => Ok(vec![raw.to_vec()]),
        [r, c] => Ok((0..*r).map(|i| raw[i * c..(i + 1) * c].to_vec()).collect()),
        other => Err(VizError::InvalidShape(format!(
            "waffle expects a vector or RxK matrix, got {other:?}"
        ))),
    }
}

/// Draw one band's blocks starting at `band_y`; return the next
/// band's top y. Counts expand into one category index per block
/// (loss/tie/win), laid row-major `COLS` wide.
fn render_band(out: &mut String, counts: &[f64], band_y: f64) -> f64 {
    let mut cats = Vec::new();
    for (cat, &c) in counts.iter().enumerate() {
        cats.extend(std::iter::repeat_n(cat, c.max(0.0).round() as usize));
    }
    for (i, &cat) in cats.iter().enumerate() {
        let x = MARGIN + (i % COLS) as f64 * (CELL + GAP);
        let y = band_y + (i / COLS) as f64 * (CELL + GAP);
        let fill = COLORS[cat.min(COLORS.len() - 1)];
        out.push_str(&format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{CELL}\" height=\"{CELL}\" rx=\"2\" fill=\"{fill}\"/>"
        ));
    }
    band_y + cats.len().div_ceil(COLS).max(1) as f64 * (CELL + GAP) + BAND_GAP
}

/// Total canvas height for all stacked bands.
fn total_height(bands: &[Vec<f64>]) -> f64 {
    let stacked: f64 = bands
        .iter()
        .map(|b| {
            let total: usize = b.iter().map(|&c| c.max(0.0).round() as usize).sum();
            total.div_ceil(COLS).max(1) as f64 * (CELL + GAP) + BAND_GAP
        })
        .sum();
    MARGIN + stacked + MARGIN - BAND_GAP
}
