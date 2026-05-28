//! Saga 32 step 002: layout + validation helpers extracted
//! from `gallery.rs::render_gallery` so the renderer fits
//! under the sw-checklist function-LOC budget.
//!
//! Three pure helpers:
//! - `validate_gallery_shapes`: confirms the input is
//!   `[N, 3, H, W]` and the optional overlay is `[N]` or
//!   `[N, K<=4]`.
//! - `compute_grid_layout`: derives rows/cols/cell sizes
//!   given N images and the canvas budget.
//! - `prepare_overlay`: unpacks the overlay tensor into a
//!   `(data, cols_per_image)` pair the row-render loop can
//!   consume directly.

use mlpl_array::DenseArray;

use super::VizError;

/// `[N, 3, H, W]` validation + optional `[N]` / `[N, K<=4]`
/// overlay shape check. Returns `(N, H, W)` on success.
pub(super) fn validate_gallery_shapes(
    data: &DenseArray,
    overlay: Option<&DenseArray>,
) -> Result<(usize, usize, usize), VizError> {
    let dims = data.shape().dims();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(VizError::InvalidShape(format!(
            "gallery expects [N, 3, H, W] shape, got {dims:?}"
        )));
    }
    let (n, src_h, src_w) = (dims[0], dims[2], dims[3]);
    if let Some(ov) = overlay {
        let od = ov.shape().dims();
        let ok = match od.len() {
            1 => od[0] == n,
            2 => od[0] == n && od[1] <= 4,
            _ => false,
        };
        if !ok {
            return Err(VizError::InvalidShape(format!(
                "gallery overlay must be [N] or [N, K<=4] with N={n}, got {od:?}"
            )));
        }
    }
    Ok((n, src_h, src_w))
}

/// Grid layout snapshot the row-render loop consumes.
pub(super) struct GridLayout {
    pub cols: usize,
    pub cell_w: f64,
    pub cell_h: f64,
    pub thumb_size: f64,
    pub label_reserve: f64,
}

/// Derive rows/cols/cell sizes from N + canvas budget. Uses a
/// square-ish grid (`ceil(sqrt(N))` columns).
pub(super) fn compute_grid_layout(
    n: usize,
    canvas_w: f64,
    canvas_h: f64,
    pad: f64,
    max_thumb: f64,
    has_overlay: bool,
) -> GridLayout {
    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    let cell_w = (canvas_w - 2.0 * pad) / cols as f64;
    let cell_h = (canvas_h - 2.0 * pad) / rows as f64;
    let label_reserve = if has_overlay { 14.0 } else { 0.0 };
    let thumb_size = cell_w.min(cell_h - label_reserve).min(max_thumb);
    GridLayout {
        cols,
        cell_w,
        cell_h,
        thumb_size,
        label_reserve,
    }
}

/// Unpack the optional overlay tensor into `(data, cols_per_image)`.
/// `cols_per_image` is `0` when no overlay, `1` for a `[N]`
/// overlay, or `od[1]` for `[N, K]`.
pub(super) fn prepare_overlay(overlay: Option<&DenseArray>) -> (Option<&[f64]>, usize) {
    match overlay {
        Some(ov) => {
            let cols = if ov.shape().dims().len() == 1 {
                1
            } else {
                ov.shape().dims()[1]
            };
            (Some(ov.data()), cols)
        }
        None => (None, 0),
    }
}
