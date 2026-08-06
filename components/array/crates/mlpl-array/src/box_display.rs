//! ASCII box-diagram renderer for `DenseArray` (APL `]display` style).
//!
//! Makes rank, shape, and depth visible at a glance: rank <= 2 renders
//! as a single framed grid; rank 3 renders one ROW of boxed matrices
//! and rank 4 an OUTER GRID of boxed matrices (APL2's DISPLAY of
//! blocks enclosed on the trailing two axes, from a flat array);
//! rank >= 5 stacks the leading-axis slices as labeled blocks. A
//! footer states rank / shape / depth. ASCII-first: frames use
//! `+ - |` only. Forward-compatible with nested arrays, which will
//! recurse to deeper frames.

use crate::dense::DenseArray;

/// Cap on element count rendered in full; larger arrays get a summary.
const MAX_CELLS: usize = 256;

/// Render `arr` as an ASCII structural box diagram plus a footer line.
#[must_use]
pub fn box_display(arr: &DenseArray) -> String {
    let depth = u8::from(arr.rank() != 0);
    let footer = format!(
        "rank {}  shape {:?}  depth {depth}",
        arr.rank(),
        arr.shape().dims()
    );
    if arr.elem_count() > MAX_CELLS {
        return format!("<{} cells; too large to box>\n{footer}", arr.elem_count());
    }
    let body = body(arr.shape().dims(), arr.data()).join("\n");
    format!("{body}\n{footer}")
}

/// Content lines for a sub-tensor: a framed grid for rank <= 2, a
/// (grid of) boxed matrices for rank 3/4, else a labeled stack of
/// the leading-axis slices (recursing one axis in).
fn body(dims: &[usize], data: &[f64]) -> Vec<String> {
    match dims.len() {
        0..=2 => framed(&grid_rows(dims, data)),
        3 => block_grid(&[1, dims[0], dims[1], dims[2]], data),
        4 => block_grid(dims, data),
        _ => {
            let stride: usize = dims[1..].iter().product();
            let mut out = Vec::new();
            for (i, chunk) in data.chunks(stride.max(1)).enumerate() {
                out.push(format!("[{i}]"));
                out.extend(body(&dims[1..], chunk));
            }
            out
        }
    }
}

/// Rank-4 [a, b, c, d] as an outer frame holding an a x b grid of
/// framed c x d matrices -- the APL2 DISPLAY shape for blocks.
fn block_grid(dims: &[usize], data: &[f64]) -> Vec<String> {
    let (outer_rows, outer_cols) = (dims[0], dims[1]);
    let inner = dims[2] * dims[3];
    let mut lines = Vec::new();
    for r in 0..outer_rows {
        let blocks: Vec<Vec<String>> = (0..outer_cols)
            .map(|c| {
                let start = (r * outer_cols + c) * inner;
                framed(&aligned_rows(dims[3], &data[start..start + inner]))
            })
            .collect();
        lines.extend(hjoin(&blocks));
    }
    framed(&lines)
}

/// Join equal-height line blocks side by side, padding each block
/// to its own width.
fn hjoin(blocks: &[Vec<String>]) -> Vec<String> {
    let height = blocks.iter().map(Vec::len).max().unwrap_or(0);
    (0..height)
        .map(|i| {
            blocks
                .iter()
                .map(|b| {
                    let width = b.iter().map(String::len).max().unwrap_or(0);
                    format!("{:<width$}", b.get(i).map_or("", String::as_str))
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// One inner block's rows, cells right-aligned to the block's
/// widest cell (block grids mix 1- and 2-digit values).
fn aligned_rows(cols: usize, data: &[f64]) -> Vec<String> {
    let cell = data.iter().map(|v| format!("{v}").len()).max().unwrap_or(1);
    data.chunks(cols.max(1))
        .map(|vals| {
            vals.iter()
                .map(|v| format!("{v:>cell$}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Unframed value rows: one row for rank 0/1, one per leading index
/// for rank 2.
fn grid_rows(dims: &[usize], data: &[f64]) -> Vec<String> {
    let cols = if dims.len() == 2 {
        dims[1].max(1)
    } else {
        data.len().max(1)
    };
    data.chunks(cols)
        .map(|vals| {
            vals.iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
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
