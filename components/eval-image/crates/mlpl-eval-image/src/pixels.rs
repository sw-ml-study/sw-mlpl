//! Saga 32 step 005: pixel-rearrangement helpers extracted
//! from `image_io.rs` so the parent module stays under the
//! sw-checklist function-count budget.
//!
//! Pure helpers, no IO. Used by the PNG / JPEG decoders in
//! `image_io.rs`.

use std::path::Path;

use crate::ImageError;

/// Convert decoded PNG bytes into an `[H, W, 3]` RGB buffer,
/// expanding grayscale / RGBA / grayscale-alpha sources as
/// needed and erroring on unsupported variants.
pub(crate) fn expand_to_rgb(
    buf: &[u8],
    h: usize,
    w: usize,
    color: png::ColorType,
    depth: png::BitDepth,
    path: &Path,
) -> Result<Vec<u8>, ImageError> {
    if depth != png::BitDepth::Eight {
        return Err(ImageError::new(format!(
            "load_images: only 8-bit PNGs supported (got {depth:?} for {})",
            path.display()
        )));
    }
    match color {
        png::ColorType::Rgb => Ok(buf.to_vec()),
        png::ColorType::Rgba => {
            let mut out = Vec::with_capacity(3 * h * w);
            for px in buf.chunks_exact(4) {
                out.extend_from_slice(&px[..3]);
            }
            Ok(out)
        }
        png::ColorType::Grayscale => Ok(grayscale_to_rgb(buf)),
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(3 * h * w);
            for px in buf.chunks_exact(2) {
                out.extend_from_slice(&[px[0], px[0], px[0]]);
            }
            Ok(out)
        }
        png::ColorType::Indexed => Err(ImageError::new(format!(
            "load_images: indexed/palette PNGs not supported in {}",
            path.display()
        ))),
    }
}

pub(crate) fn grayscale_to_rgb(buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len() * 3);
    for &v in buf {
        out.extend_from_slice(&[v, v, v]);
    }
    out
}

/// Bilinear resize from `(src_h, src_w)` to `(dst_h, dst_w)`.
/// Input + output are `[H, W, 3]` row-major u8 RGB. The
/// interpolation matches OpenCV's `INTER_LINEAR` half-pixel
/// convention so MLPL's resize agrees with the reference
/// notebook on the Oxford-IIIT Pet samples.
pub(crate) fn bilinear_resize_rgb_hwc(
    src: &[u8],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_h * dst_w * 3];
    let scale_y = src_h as f64 / dst_h as f64;
    let scale_x = src_w as f64 / dst_w as f64;
    for y in 0..dst_h {
        let sy = (y as f64 + 0.5) * scale_y - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let dy = (sy - y0 as f64).clamp(0.0, 1.0);
        for x in 0..dst_w {
            let sx = (x as f64 + 0.5) * scale_x - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let dx = (sx - x0 as f64).clamp(0.0, 1.0);
            for c in 0..3 {
                let p00 = f64::from(src[(y0 * src_w + x0) * 3 + c]);
                let p01 = f64::from(src[(y0 * src_w + x1) * 3 + c]);
                let p10 = f64::from(src[(y1 * src_w + x0) * 3 + c]);
                let p11 = f64::from(src[(y1 * src_w + x1) * 3 + c]);
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v = (top * (1.0 - dy) + bot * dy).round().clamp(0.0, 255.0);
                out[(y * dst_w + x) * 3 + c] = v as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_to_rgb_triples_bytes() {
        let g = vec![0u8, 128, 255];
        let rgb = grayscale_to_rgb(&g);
        assert_eq!(rgb, vec![0, 0, 0, 128, 128, 128, 255, 255, 255]);
    }

    #[test]
    fn bilinear_resize_identity_when_dims_match() {
        let src: Vec<u8> = (0..(2 * 2 * 3)).map(|i| i as u8 * 10).collect();
        let out = bilinear_resize_rgb_hwc(&src, 2, 2, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn bilinear_resize_upsample_smooths() {
        // 2x2 solid colors -> 4x4 stays solid in the interior.
        let src = vec![255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255]; // 2x2 [red, red, blue, blue]
        let out = bilinear_resize_rgb_hwc(&src, 2, 2, 4, 4);
        // Top-left pixel should still be red-dominant.
        assert!(out[0] >= 200, "top-left should stay red, got {}", out[0]);
        assert_eq!(out[2], 0, "top-left blue channel should stay 0");
        // Bottom-right pixel should still be blue-dominant.
        let br = (3 * 4 + 3) * 3;
        assert_eq!(out[br], 0, "bottom-right red channel should stay 0");
        assert!(out[br + 2] >= 200, "bottom-right should stay blue");
    }
}
