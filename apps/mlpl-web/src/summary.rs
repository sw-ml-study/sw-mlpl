//! Numeric output summarization for the REPL display.
//!
//! Large numeric outputs (long vectors / matrices) are parsed back into
//! `f64`s so the UI can show a one-line summary (shape + stats) with the
//! raw values hidden inside a collapsible `<details>` element.

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
fn parse_numeric_grid(output: &str) -> Option<(Vec<f64>, usize, Option<usize>)> {
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

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}
fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn median(v: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
fn std(v: &[f64]) -> f64 {
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    var.sqrt()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_output_is_not_summarized() {
        assert!(summarize("42").is_none());
        assert!(summarize("1 2 3").is_none());
        assert!(summarize("1 2 3\n4 5 6").is_none());
    }

    #[test]
    fn long_vector_is_summarized_as_1d() {
        let out = (0..20).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
        // long enough via chars? 20 numbers fit under 200 chars, so pad.
        let out = format!("{out}\n{out}\n{out}\n{out}\n{out}\n{out}\n{out}\n{out}\n{out}");
        let s = summarize(&out).expect("should summarize");
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 19.0);
    }

    #[test]
    fn uniform_matrix_reports_rows_cols() {
        let out = (0..9)
            .map(|i| format!("{i} {i} {i} {i} {i} {i} {i} {i} {i} {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let s = summarize(&out).expect("should summarize");
        assert!(s.shape.contains("9x10"), "shape was {}", s.shape);
    }

    #[test]
    fn non_numeric_output_is_not_summarized() {
        let out = "line one\nline two\nline three\nline four\nline five\n\
                   line six\nline seven\nline eight\nline nine\nline ten";
        assert!(summarize(out).is_none());
    }

    #[test]
    fn stats_are_correct_for_known_vector() {
        // 1..=20, long enough to trigger via line count.
        let rows: Vec<String> = (1..=20).map(|i| i.to_string()).collect();
        let out = rows.join("\n");
        let s = summarize(&out).expect("should summarize");
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 20.0);
        assert!((s.mean - 10.5).abs() < 1e-9);
        assert!((s.median - 10.5).abs() < 1e-9);
    }
}
