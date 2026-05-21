//! Gallery rendering: turn an `[N, 3, H, W]` image batch
//! into an SVG grid of thumbnails. Saga 29 step 010.
//!
//! Each batch entry is treated as a single RGB image with
//! channel-first layout (`[3, H, W]` -- the same layout the
//! `load_preloaded("pets_tiny")` fixture and `load_images`
//! builtin produce). Values are expected in `[-1, 1]`
//! (the normalized space) but values outside that range are
//! clamped at render time so a stray un-normalized tensor
//! still renders, just with all the colors saturated.
//!
//! The thumbnails are downsampled to fit the gallery cell
//! size via block averaging -- so a `[20, 3, 64, 64]`
//! pets_tiny slice renders as twenty ~40x40 thumbnails in
//! the 400x300 SVG canvas without emitting 80K `<rect>`
//! elements.

use mlpl_array::DenseArray;

use super::{H, PAD, VizError, W, write_svg_close, write_svg_open};

/// Maximum displayed thumbnail size (in SVG units). The grid
/// scales thumbnails DOWN to fit the canvas; thumbnails are
/// not stretched UP beyond their source resolution.
const MAX_THUMB: f64 = 60.0;

/// Render an `[N, 3, H, W]` batch as a grid of RGB
/// thumbnails. Returns a self-contained SVG string.
///
/// `overlay` (Saga 29 step 011) is optional. Pass
/// `Some([N])` to label each thumbnail with one integer
/// value (e.g. predictions only), or `Some([N, K])` (K up
/// to 4) for multiple values per thumbnail (e.g. actual /
/// predicted as a 2-column matrix). Values render as
/// integers separated by `/`; class-name mapping is the
/// caller's job.
pub fn render_gallery(data: &DenseArray, overlay: Option<&DenseArray>) -> Result<String, VizError> {
    let dims = data.shape().dims();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(VizError::InvalidShape(format!(
            "gallery expects [N, 3, H, W] shape, got {dims:?}"
        )));
    }
    let n = dims[0];
    let src_h = dims[2];
    let src_w = dims[3];
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
    let mut out = String::new();
    write_svg_open(&mut out);
    if n == 0 || src_h == 0 || src_w == 0 {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    let cell_w = (W - 2.0 * PAD) / cols as f64;
    let cell_h = (H - 2.0 * PAD) / rows as f64;
    let label_reserve = if overlay.is_some() { 14.0 } else { 0.0 };
    let thumb_size = cell_w.min(cell_h - label_reserve).min(MAX_THUMB);
    let (thumb_h_px, thumb_w_px) = (
        src_h.min(thumb_size as usize),
        src_w.min(thumb_size as usize),
    );
    let raw = data.data();
    let stride_n = 3 * src_h * src_w;
    let (ov_data, ov_cols) = match overlay {
        Some(ov) => {
            let cols_per = if ov.shape().dims().len() == 1 {
                1
            } else {
                ov.shape().dims()[1]
            };
            (Some(ov.data()), cols_per)
        }
        None => (None, 0),
    };
    for idx in 0..n {
        let col = idx % cols;
        let row = idx / cols;
        let cell_x = PAD + cell_w * col as f64 + (cell_w - thumb_size) * 0.5;
        let cell_y = PAD + cell_h * row as f64 + (cell_h - thumb_size - label_reserve) * 0.5;
        let img = &raw[idx * stride_n..(idx + 1) * stride_n];
        render_thumbnail(
            &mut out,
            img,
            (src_h, src_w),
            (thumb_h_px, thumb_w_px),
            (cell_x, cell_y),
            thumb_size,
        );
        if let Some(values) = ov_data {
            render_overlay_label(
                &mut out,
                values,
                idx,
                ov_cols,
                cell_x,
                cell_y + thumb_size + 1.0,
                thumb_size,
            );
        }
    }
    write_svg_close(&mut out);
    Ok(out)
}

/// Render the `K` label values for thumbnail `idx` as a
/// small text caption centered under the thumbnail.
fn render_overlay_label(
    out: &mut String,
    values: &[f64],
    idx: usize,
    ov_cols: usize,
    x: f64,
    y: f64,
    thumb_size: f64,
) {
    let mut parts: Vec<String> = Vec::with_capacity(ov_cols);
    for c in 0..ov_cols {
        let v = values[idx * ov_cols + c];
        parts.push(format!("{v:.0}"));
    }
    let text = parts.join("/");
    let cx = x + thumb_size * 0.5;
    out.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{y:.1}\" fill=\"#cdd6f4\" font-size=\"9\" \
         font-family=\"monospace\" text-anchor=\"middle\" dominant-baseline=\"hanging\">{text}</text>"
    ));
}

fn render_thumbnail(
    out: &mut String,
    img: &[f64],
    src: (usize, usize),
    thumb_px: (usize, usize),
    cell: (f64, f64),
    thumb_size: f64,
) {
    let (src_h, src_w) = src;
    let (thumb_h_px, thumb_w_px) = thumb_px;
    let (cell_x, cell_y) = cell;
    let pixel_w = thumb_size / thumb_w_px as f64;
    let pixel_h = thumb_size / thumb_h_px as f64;
    let block_h = src_h as f64 / thumb_h_px as f64;
    let block_w = src_w as f64 / thumb_w_px as f64;
    let chan_stride = src_h * src_w;
    for ty in 0..thumb_h_px {
        for tx in 0..thumb_w_px {
            let y0 = (ty as f64 * block_h) as usize;
            let y1 = (((ty as f64 + 1.0) * block_h) as usize)
                .max(y0 + 1)
                .min(src_h);
            let x0 = (tx as f64 * block_w) as usize;
            let x1 = (((tx as f64 + 1.0) * block_w) as usize)
                .max(x0 + 1)
                .min(src_w);
            let mut sum = [0.0_f64; 3];
            for c in 0..3 {
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        sum[c] += img[c * chan_stride + sy * src_w + sx];
                    }
                }
            }
            let count = ((y1 - y0) * (x1 - x0)) as f64;
            let inv = if count > 0.0 { 1.0 / count } else { 1.0 };
            let r = norm_to_u8(sum[0] * inv);
            let g = norm_to_u8(sum[1] * inv);
            let b = norm_to_u8(sum[2] * inv);
            let px = cell_x + tx as f64 * pixel_w;
            let py = cell_y + ty as f64 * pixel_h;
            out.push_str(&format!(
                "<rect x=\"{px:.1}\" y=\"{py:.1}\" width=\"{pixel_w:.2}\" height=\"{pixel_h:.2}\" fill=\"rgb({r},{g},{b})\"/>"
            ));
        }
    }
}

/// Map a `[-1, 1]`-normalized pixel value to a `[0, 255]`
/// u8 channel. Values outside the range clamp instead of
/// wrapping so non-pets_tiny data still renders.
fn norm_to_u8(v: f64) -> u8 {
    let unit = ((v + 1.0) * 127.5).clamp(0.0, 255.0);
    unit.round() as u8
}

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
    if !src_h.is_multiple_of(side) || !src_w.is_multiple_of(side) {
        return Err(VizError::InvalidShape(format!(
            "attention_overlay image {src_h}x{src_w} not divisible by sqrt(patches)={side}"
        )));
    }
    Ok((src_h, src_w, heads, patches, side))
}
