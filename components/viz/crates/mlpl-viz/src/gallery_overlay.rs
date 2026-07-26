//! Attention-overlay rendering: a translucent viridis heat grid over
//! image thumbnails, one tile per head. Split from `gallery.rs`
//! (tech-debt spike step 001).

use mlpl_array::DenseArray;

use crate::gallery::{render_overlay_label, render_thumbnail};
use mlpl_viz_core::{H, PAD, VizError, W, write_svg_close, write_svg_open};

/// Render an `[3, H, W]` image with an attention heatmap
/// overlaid as translucent patch-sized rectangles.
///
/// `attn` shape is either `[P]` (single head) or `[heads, P]`
/// (multi-head). `P` must be a perfect square; `H` and `W`
/// must be divisible by `sqrt(P)`. Output is a single tile
/// for single-head, or a `ceil(sqrt(heads))`-column grid of
/// tiles for multi-head, with a head index label under each.
pub fn render_attention_overlay(
    image: &DenseArray,
    attn: Option<&DenseArray>,
) -> Result<String, VizError> {
    let attn = attn.ok_or_else(|| {
        VizError::InvalidShape(
            "attention_overlay requires a third argument (attention weights [P] or [heads, P])"
                .into(),
        )
    })?;
    let (src_h, src_w, heads, patches, side) = parse_overlay_shapes(image, attn)?;
    let mut out = String::new();
    write_svg_open(&mut out);
    let cols = (heads as f64).sqrt().ceil() as usize;
    let rows = heads.div_ceil(cols);
    let cell_w = (W - 2.0 * PAD) / cols as f64;
    let cell_h = (H - 2.0 * PAD) / rows as f64;
    let label_reserve = if heads > 1 { 12.0 } else { 0.0 };
    let tile = cell_w.min(cell_h - label_reserve).min(MAX_TILE);
    let (img_data, attn_data) = (image.data(), attn.data());
    let (thumb_h, thumb_w) = (src_h.min(tile as usize), src_w.min(tile as usize));
    for h in 0..heads {
        let (col, row) = (h % cols, h / cols);
        let cell_x = PAD + cell_w * col as f64 + (cell_w - tile) * 0.5;
        let cell_y = PAD + cell_h * row as f64 + (cell_h - tile - label_reserve) * 0.5;
        render_thumbnail(
            &mut out,
            img_data,
            (src_h, src_w),
            (thumb_h, thumb_w),
            (cell_x, cell_y),
            tile,
        );
        let slab = &attn_data[h * patches..(h + 1) * patches];
        render_overlay_grid(&mut out, slab, side, cell_x, cell_y, tile);
        if heads > 1 {
            let lbl_y = cell_y + tile + 1.0;
            render_overlay_label(&mut out, &[h as f64], 0, 1, cell_x, lbl_y, tile);
        }
    }
    write_svg_close(&mut out);
    Ok(out)
}

const MAX_TILE: f64 = 110.0;

/// Translucent viridis rect per patch, scaled by attention magnitude.
fn render_overlay_grid(
    out: &mut String,
    head_attn: &[f64],
    side: usize,
    cell_x: f64,
    cell_y: f64,
    tile: f64,
) {
    let viridis = |t: f64| -> (u8, u8, u8) {
        const STOPS: [(f64, f64, f64); 3] = [
            (68.0, 1.0, 84.0),
            (33.0, 145.0, 140.0),
            (253.0, 231.0, 37.0),
        ];
        let (a, b, frac) = if t < 0.5 {
            (STOPS[0], STOPS[1], t * 2.0)
        } else {
            (STOPS[1], STOPS[2], (t - 0.5) * 2.0)
        };
        let lerp = |x: f64, y: f64| (x + (y - x) * frac).round() as u8;
        (lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2))
    };
    let patch = tile / side as f64;
    let lo = head_attn.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = head_attn.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = if (hi - lo).abs() < f64::EPSILON {
        1.0
    } else {
        hi - lo
    };
    for r in 0..side {
        for c in 0..side {
            let t = ((head_attn[r * side + c] - lo) / span).clamp(0.0, 1.0);
            let (red, green, blue) = viridis(t);
            let alpha = 0.35 + 0.40 * t;
            let (px, py) = (cell_x + c as f64 * patch, cell_y + r as f64 * patch);
            out.push_str(&format!(
                "<rect x=\"{px:.2}\" y=\"{py:.2}\" width=\"{patch:.2}\" height=\"{patch:.2}\" \
                 fill=\"rgb({red},{green},{blue})\" fill-opacity=\"{alpha:.2}\"/>"
            ));
        }
    }
}

/// Validate `image` (`[3, H, W]`) and `attn` (`[P]` or `[heads, P]`)
/// shapes and return `(src_h, src_w, heads, patches, side)`.
fn parse_overlay_shapes(
    image: &DenseArray,
    attn: &DenseArray,
) -> Result<(usize, usize, usize, usize, usize), VizError> {
    let img_dims = image.shape().dims();
    if img_dims.len() != 3 || img_dims[0] != 3 {
        return Err(VizError::InvalidShape(format!(
            "attention_overlay image must be [3, H, W], got {img_dims:?}"
        )));
    }
    let (src_h, src_w) = (img_dims[1], img_dims[2]);
    let (heads, patches, side) = parse_attn_shape(attn)?;
    if !src_h.is_multiple_of(side) || !src_w.is_multiple_of(side) {
        return Err(VizError::InvalidShape(format!(
            "attention_overlay image {src_h}x{src_w} not divisible by sqrt(patches)={side}"
        )));
    }
    Ok((src_h, src_w, heads, patches, side))
}

/// `attn` must be `[P]` or `[heads, P]` with `P` a perfect square;
/// returns `(heads, patches, side)`.
fn parse_attn_shape(attn: &DenseArray) -> Result<(usize, usize, usize), VizError> {
    let attn_dims = attn.shape().dims();
    let (heads, patches) = match attn_dims.len() {
        1 => (1, attn_dims[0]),
        2 => (attn_dims[0], attn_dims[1]),
        _ => {
            return Err(VizError::InvalidShape(format!(
                "attention_overlay attn must be [P] or [heads, P], got {attn_dims:?}"
            )));
        }
    };
    let side = (patches as f64).sqrt().round() as usize;
    if side * side != patches {
        return Err(VizError::InvalidShape(format!(
            "attention_overlay patch count {patches} is not a perfect square"
        )));
    }
    Ok((heads, patches, side))
}
