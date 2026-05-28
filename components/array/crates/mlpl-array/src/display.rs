//! Display formatting for DenseArray.

use crate::dense::DenseArray;
use std::fmt;

// Saga 29 step 009 follow-up: when a high-rank or
// many-element tensor would otherwise dump its entire flat
// data via `{:?}`, fall back to a shape-only summary.
// `pets_tiny.X` is `[200, 3, 64, 64]` = 2,457,600 floats;
// the old impl emitted ~50 MB of text to the REPL, which
// made the browser DOM crawl. Threshold of 64 elements
// keeps small rank-3+ debug tensors fully visible while
// truncating real-image / batch tensors.
const ARRAY_DISPLAY_FULL_LIMIT: usize = 64;

impl fmt::Display for DenseArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.elem_count() == 0 {
            return write!(f, "[]");
        }
        match self.rank() {
            0 => write!(f, "{}", self.data()[0]),
            1 if self.elem_count() <= ARRAY_DISPLAY_FULL_LIMIT => write_row(f, self.data()),
            1 => write_truncated(f, self.shape().dims(), self.data()),
            2 if self.elem_count() <= ARRAY_DISPLAY_FULL_LIMIT => {
                let cols = self.shape().dims()[1];
                for (i, row) in self.data().chunks(cols).enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write_row(f, row)?;
                }
                Ok(())
            }
            2 => write_truncated(f, self.shape().dims(), self.data()),
            _ if self.elem_count() <= ARRAY_DISPLAY_FULL_LIMIT => {
                write!(f, "{:?}", self.data())
            }
            _ => write_truncated(f, self.shape().dims(), self.data()),
        }
    }
}

/// Short, lossy summary for tensors too big to print in full.
/// Shows shape + element count + the first three elements so
/// the user gets a sanity-check on the values without paying
/// for a megabyte of text. The full data is still reachable
/// via `data()` from Rust callers or via index/slice ops at
/// the language level.
fn write_truncated(f: &mut fmt::Formatter<'_>, dims: &[usize], data: &[f64]) -> fmt::Result {
    let n = data.len();
    let preview = data.iter().take(3).copied().collect::<Vec<_>>();
    let preview_strs: Vec<String> = preview.iter().map(|v| format!("{v}")).collect();
    write!(
        f,
        "<DenseArray shape={dims:?} elems={n} first=[{}, ...]>",
        preview_strs.join(", ")
    )
}

/// Format a single row of f64 values, space-separated.
fn write_row(f: &mut fmt::Formatter<'_>, values: &[f64]) -> fmt::Result {
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            write!(f, " ")?;
        }
        write!(f, "{v}")?;
    }
    Ok(())
}
