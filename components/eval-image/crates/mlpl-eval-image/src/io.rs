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

use crate::ImageError;

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
pub fn decode_and_resize(
    path: &Path,
    target_h: usize,
    target_w: usize,
) -> Result<Vec<f64>, ImageError> {
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
) -> Result<Vec<u8>, ImageError> {
    let bytes = fs::read(path).map_err(|e| {
        ImageError::new(format!("load_images: cannot read {}: {e}", path.display()))
    })?;
    let (rgb, src_h, src_w) = decode_bytes(&bytes, path)?;
    Ok(crate::pixels::bilinear_resize_rgb_hwc(
        &rgb, src_h, src_w, target_h, target_w,
    ))
}

/// Sniff magic bytes and dispatch to the matching decoder.
/// Returns the decoded image as `[H, W, 3]` row-major
/// `(y, x, channel)` u8 RGB plus the source dimensions.
fn decode_bytes(bytes: &[u8], path: &Path) -> Result<(Vec<u8>, usize, usize), ImageError> {
    if bytes.starts_with(PNG_MAGIC) {
        decode_png(bytes, path)
    } else if bytes.starts_with(JPEG_MAGIC) {
        decode_jpeg(bytes, path)
    } else {
        Err(ImageError::new(format!(
            "load_images: {} is not a recognized PNG or JPEG \
             (got magic bytes {:02X?})",
            path.display(),
            &bytes[..bytes.len().min(4)]
        )))
    }
}

fn decode_png(bytes: &[u8], path: &Path) -> Result<(Vec<u8>, usize, usize), ImageError> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| {
        ImageError::new(format!(
            "load_images: PNG header error in {}: {e}",
            path.display()
        ))
    })?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| {
        ImageError::new(format!(
            "load_images: PNG decode error in {}: {e}",
            path.display()
        ))
    })?;
    buf.truncate(info.buffer_size());
    let h = info.height as usize;
    let w = info.width as usize;
    let rgb = crate::pixels::expand_to_rgb(&buf, h, w, info.color_type, info.bit_depth, path)?;
    Ok((rgb, h, w))
}

fn decode_jpeg(bytes: &[u8], path: &Path) -> Result<(Vec<u8>, usize, usize), ImageError> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    let pixels = decoder.decode().map_err(|e| {
        ImageError::new(format!(
            "load_images: JPEG decode error in {}: {e}",
            path.display()
        ))
    })?;
    let info = decoder.info().ok_or_else(|| {
        ImageError::new(format!(
            "load_images: JPEG missing info for {}",
            path.display()
        ))
    })?;
    let h = info.height as usize;
    let w = info.width as usize;
    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => crate::pixels::grayscale_to_rgb(&pixels),
        other => {
            return Err(ImageError::new(format!(
                "load_images: unsupported JPEG pixel format {other:?} in {}",
                path.display()
            )));
        }
    };
    Ok((rgb, h, w))
}
