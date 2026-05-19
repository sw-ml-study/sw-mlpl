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
pub fn render_gallery(data: &DenseArray) -> Result<String, VizError> {
    let dims = data.shape().dims();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(VizError::InvalidShape(format!(
            "gallery expects [N, 3, H, W] shape, got {dims:?}"
        )));
    }
    let n = dims[0];
    let src_h = dims[2];
    let src_w = dims[3];
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
    let thumb_size = cell_w.min(cell_h).min(MAX_THUMB);
    let (thumb_h_px, thumb_w_px) = (
        src_h.min(thumb_size as usize),
        src_w.min(thumb_size as usize),
    );
    let raw = data.data();
    let stride_n = 3 * src_h * src_w;
    for idx in 0..n {
        let col = idx % cols;
        let row = idx / cols;
        let cell_x = PAD + cell_w * col as f64 + (cell_w - thumb_size) * 0.5;
        let cell_y = PAD + cell_h * row as f64 + (cell_h - thumb_size) * 0.5;
        let img = &raw[idx * stride_n..(idx + 1) * stride_n];
        render_thumbnail(
            &mut out, img, src_h, src_w, thumb_h_px, thumb_w_px, cell_x, cell_y, thumb_size,
        );
    }
    write_svg_close(&mut out);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn render_thumbnail(
    out: &mut String,
    img: &[f64],
    src_h: usize,
    src_w: usize,
    thumb_h_px: usize,
    thumb_w_px: usize,
    cell_x: f64,
    cell_y: f64,
    thumb_size: f64,
) {
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
            let mut count = 0.0_f64;
            for c in 0..3 {
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        sum[c] += img[c * chan_stride + sy * src_w + sx];
                    }
                }
            }
            count += ((y1 - y0) * (x1 - x0)) as f64;
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
