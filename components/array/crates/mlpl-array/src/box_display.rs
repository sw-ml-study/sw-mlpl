//! ASCII box-diagram renderer for `DenseArray` (APL `]display` style).
//!
//! Makes rank, shape, and depth visible at a glance: rank <= 2 renders
//! as a single framed grid; rank >= 3 stacks the leading-axis slices as
//! labeled blocks. A footer states rank / shape / depth. ASCII-first:
//! the frame uses `+ - |` only. Forward-compatible with nested arrays
//! (staging plan Stage 6), which will recurse to deeper frames.

use crate::dense::DenseArray;

/// Cap on element count rendered in full; larger arrays get a summary.
const MAX_CELLS: usize = 256;

/// Render `arr` as an ASCII structural box diagram plus a footer line.
#[must_use]
pub fn box_display(arr: &DenseArray) -> String {
    let footer = footer(arr);
    if arr.elem_count() > MAX_CELLS {
        return format!("<{} cells; too large to box>\n{footer}", arr.elem_count());
    }
    let body = body(arr.shape().dims(), arr.data()).join("\n");
    format!("{body}\n{footer}")
}

fn footer(arr: &DenseArray) -> String {
    let depth = u8::from(arr.rank() != 0);
    format!(
        "rank {}  shape {:?}  depth {depth}",
        arr.rank(),
        arr.shape().dims()
    )
}

/// Content lines for a sub-tensor: a framed grid for rank <= 2, else a
/// labeled stack of the leading-axis slices (recursing one axis in).
fn body(dims: &[usize], data: &[f64]) -> Vec<String> {
    if dims.len() <= 2 {
        return framed(&grid_rows(dims, data));
    }
    let stride: usize = dims[1..].iter().product();
    let mut out = Vec::new();
    for (i, chunk) in data.chunks(stride.max(1)).enumerate() {
        out.push(format!("[{i}]"));
        out.extend(body(&dims[1..], chunk));
    }
    out
}

/// Unframed value rows: one row for rank 0/1, one per leading index
/// for rank 2.
fn grid_rows(dims: &[usize], data: &[f64]) -> Vec<String> {
    let cols = if dims.len() == 2 {
        dims[1].max(1)
    } else {
        data.len().max(1)
    };
    data.chunks(cols).map(join_row).collect()
}

fn join_row(vals: &[f64]) -> String {
    vals.iter()
        .map(|v| format!("{v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Wrap content rows in an ASCII `+---+ / | .. |` frame padded to the
/// widest row.
fn framed(rows: &[String]) -> Vec<String> {
    let width = rows.iter().map(String::len).max().unwrap_or(0);
    let border = format!("+{}+", "-".repeat(width + 2));
    let mut out = vec![border.clone()];
    out.extend(rows.iter().map(|row| format!("| {row:<width$} |")));
    out.push(border);
    out
}
