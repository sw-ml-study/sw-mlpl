//! PNG / JPEG decode + resize + normalize for `load_images`.
//! Saga 29 step 003.
//!
//! Native-only: this whole module is gated behind the
//! `image-io` Cargo feature so the WASM target does not pull
//! in `png` / `jpeg-decoder`. The WASM REPL exercises image
//! data via `load_preloaded("pets_tiny")` (pre-decoded bytes
//! shipped in the binary) rather than live decode.
//!
//! Decoder choice: `png` + `jpeg-decoder` separately rather
//! than `image-rs` (smaller dep footprint). Dispatch on the
//! file's leading magic bytes -- the two formats are
//! unambiguous from the first 4 bytes.
//!
//! Output format: `[3, H, W]` u8 RGB in row-major
//! `(channel, y, x)` order, normalized to f64 in `[-1, 1]`
//! by `v / 127.5 - 1.0`. Stacking N of these gives the
//! `[N, 3, H, W]` shape with `[batch, channel, y, x]` axis
//! labels that the ViT model expects.

#![cfg(feature = "image-io")]

use std::fs;
use std::io::Cursor;
use std::path::Path;

use crate::error::EvalError;

/// PNG magic bytes: `89 50 4E 47 0D 0A 1A 0A`. Checking the
/// first 4 is sufficient to distinguish from JPEG.
const PNG_MAGIC: &[u8] = &[0x89, 0x50, 0x4E, 0x47];

/// JPEG magic bytes: `FF D8 FF`. (Fourth byte varies: `DB`,
/// `E0`, `E1`, etc., depending on JFIF / EXIF marker.)
const JPEG_MAGIC: &[u8] = &[0xFF, 0xD8, 0xFF];

/// Read an image file off disk, decode to u8 RGB, resize to
/// `(target_h, target_w)`, normalize to f64 in `[-1, 1]`, and
/// return the result in `[3, H, W]` row-major
/// `(channel, y, x)` order. The caller stacks N of these
/// into the final `[N, 3, H, W]` array.
pub(crate) fn decode_and_resize(
    path: &Path,
    target_h: usize,
    target_w: usize,
) -> Result<Vec<f64>, EvalError> {
    let resized = decode_and_resize_u8(path, target_h, target_w)?;
    // Reorder HWC -> CHW and normalize in the same pass.
    let mut out = vec![0f64; 3 * target_h * target_w];
    for c in 0..3 {
        for y in 0..target_h {
            for x in 0..target_w {
                let src = (y * target_w + x) * 3 + c;
                let dst = c * target_h * target_w + y * target_w + x;
                out[dst] = f64::from(resized[src]) / 127.5 - 1.0;
            }
        }
    }
    Ok(out)
}

/// Decode a PNG / JPEG file to u8 RGB in `[H, W, 3]`
/// row-major `(y, x, channel)` order, resized to
/// `(target_h, target_w)` via bilinear interpolation.
/// Public for the offline `pets_tiny` fixture builder; the
/// runtime path goes through `decode_and_resize` and
/// normalizes to f64.
pub fn decode_and_resize_u8(
    path: &Path,
    target_h: usize,
    target_w: usize,
) -> Result<Vec<u8>, EvalError> {
    let bytes = fs::read(path).map_err(|e| {
        EvalError::Unsupported(format!("load_images: cannot read {}: {e}", path.display()))
    })?;
    let (rgb, src_h, src_w) = decode_bytes(&bytes, path)?;
    Ok(bilinear_resize_rgb_hwc(
        &rgb, src_h, src_w, target_h, target_w,
    ))
}

/// Sniff magic bytes and dispatch to the matching decoder.
/// Returns the decoded image as `[H, W, 3]` row-major
/// `(y, x, channel)` u8 RGB plus the source dimensions.
fn decode_bytes(bytes: &[u8], path: &Path) -> Result<(Vec<u8>, usize, usize), EvalError> {
    if bytes.starts_with(PNG_MAGIC) {
        decode_png(bytes, path)
    } else if bytes.starts_with(JPEG_MAGIC) {
        decode_jpeg(bytes, path)
    } else {
        Err(EvalError::Unsupported(format!(
            "load_images: {} is not a recognized PNG or JPEG \
             (got magic bytes {:02X?})",
            path.display(),
            &bytes[..bytes.len().min(4)]
        )))
    }
}

fn decode_png(bytes: &[u8], path: &Path) -> Result<(Vec<u8>, usize, usize), EvalError> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| {
        EvalError::Unsupported(format!(
            "load_images: PNG header error in {}: {e}",
            path.display()
        ))
    })?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| {
        EvalError::Unsupported(format!(
            "load_images: PNG decode error in {}: {e}",
            path.display()
        ))
    })?;
    buf.truncate(info.buffer_size());
    let h = info.height as usize;
    let w = info.width as usize;
    let rgb = expand_to_rgb(&buf, h, w, info.color_type, info.bit_depth, path)?;
    Ok((rgb, h, w))
}

fn decode_jpeg(bytes: &[u8], path: &Path) -> Result<(Vec<u8>, usize, usize), EvalError> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    let pixels = decoder.decode().map_err(|e| {
        EvalError::Unsupported(format!(
            "load_images: JPEG decode error in {}: {e}",
            path.display()
        ))
    })?;
    let info = decoder.info().ok_or_else(|| {
        EvalError::Unsupported(format!(
            "load_images: JPEG missing info for {}",
            path.display()
        ))
    })?;
    let h = info.height as usize;
    let w = info.width as usize;
    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => grayscale_to_rgb(&pixels),
        other => {
            return Err(EvalError::Unsupported(format!(
                "load_images: unsupported JPEG pixel format {other:?} in {}",
                path.display()
            )));
        }
    };
    Ok((rgb, h, w))
}

/// Expand a PNG buffer (grayscale, RGB, RGBA, palette) into
/// 8-bit RGB. Drops alpha; refuses palette / 16-bit (rare
/// in the Oxford-IIIT Pet dataset).
fn expand_to_rgb(
    buf: &[u8],
    h: usize,
    w: usize,
    color: png::ColorType,
    depth: png::BitDepth,
    path: &Path,
) -> Result<Vec<u8>, EvalError> {
    if depth != png::BitDepth::Eight {
        return Err(EvalError::Unsupported(format!(
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
        png::ColorType::Indexed => Err(EvalError::Unsupported(format!(
            "load_images: indexed/palette PNGs not supported in {}",
            path.display()
        ))),
    }
}

fn grayscale_to_rgb(buf: &[u8]) -> Vec<u8> {
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
fn bilinear_resize_rgb_hwc(
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
