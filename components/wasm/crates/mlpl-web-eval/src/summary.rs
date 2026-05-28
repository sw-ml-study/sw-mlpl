//! Numeric output summarization for the REPL display.
//!
//! Large numeric outputs (long vectors / matrices) are parsed back into
//! `f64`s so the UI can show a one-line summary (shape + stats) with the
//! raw values hidden inside a collapsible `<details>` element.

use crate::summary_stats::{max, mean, median, min, std};

const LINE_THRESHOLD: usize = 8;
const CHAR_THRESHOLD: usize = 200;
const MIN_NUMERIC_COUNT: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct NumericSummary {
    pub shape: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std: f64,
}

/// Decide whether an output string is long enough to deserve collapsing,
/// and if so, whether it is a numeric array we can summarize.
///
/// Returns `Some(summary)` when the output should be rendered with a
/// summary line + `<details>` accordion. Returns `None` when the output
/// is short, non-numeric, or too small to summarize meaningfully.
pub fn summarize(output: &str) -> Option<NumericSummary> {
    let line_count = output.lines().count();
    if line_count <= LINE_THRESHOLD && output.len() <= CHAR_THRESHOLD {
        return None;
    }

    let (values, rows, cols) = parse_numeric_grid(output)?;
    if values.len() < MIN_NUMERIC_COUNT {
        return None;
    }

    let shape = match (rows, cols) {
        (r, Some(c)) => format!("{}x{} ({} values)", r, c, values.len()),
        (_, None) => format!("[{}]", values.len()),
    };

    Some(NumericSummary {
        shape,
        min: min(&values),
        max: max(&values),
        mean: mean(&values),
        median: median(&values),
        std: std(&values),
    })
}

/// Parse a whitespace-separated numeric grid. Returns the flat values,
/// the number of rows, and `Some(cols)` when every row has the same
/// column count (i.e. a proper 2D matrix).
pub(crate) fn parse_numeric_grid(output: &str) -> Option<(Vec<f64>, usize, Option<usize>)> {
    let mut values = Vec::new();
    let mut row_widths = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut width = 0usize;
        for tok in trimmed.split_whitespace() {
            let v: f64 = tok.parse().ok()?;
            values.push(v);
            width += 1;
        }
        row_widths.push(width);
    }
    if values.is_empty() || row_widths.is_empty() {
        return None;
    }
    let rows = row_widths.len();
    let first = row_widths[0];
    let uniform = row_widths.iter().all(|w| *w == first);
    let cols = if uniform && rows > 1 && first > 1 {
        Some(first)
    } else {
        None
    };
    Some((values, rows, cols))
}

/// Format a single f64 for the summary line (4 significant digits).
pub fn fmt_stat(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let abs = x.abs();
    if !(1e-3..1e5).contains(&abs) {
        format!("{x:.3e}")
    } else {
        format!("{x:.4}")
    }
}
