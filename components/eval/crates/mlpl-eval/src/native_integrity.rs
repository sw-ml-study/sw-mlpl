//! MLPB header + integrity for the typed-native codec. `read_header`
//! validates magic / version / payload length and accepts BOTH v1
//! (no checksum) and v2 (a trailing CRC32 over the payload).
//! `verify_checksum` recomputes the v2 CRC32 and errors on
//! mismatch; `crc32` is the bitwise IEEE (poly 0xEDB88320) checksum
//! -- no lookup table, no extra dependency. All failures are err
//! Results (never a panic), keeping the codec fail-soft.

use crate::native_cursor::{read_u8, read_u32};
use crate::native_encode::{MAGIC, VERSION};

/// Validate the MLPB header and return
/// `(version, payload_start, payload_len)`. Accepts v1 (no
/// checksum) and v2 (a 4-byte CRC32 trailer after the payload).
pub(crate) fn read_header(bytes: &[u8]) -> Result<(u8, usize, usize), String> {
    let magic = bytes.get(0..4).ok_or("truncated: missing MLPB header")?;
    if magic != MAGIC {
        return Err("bad magic: not a native (MLPB) buffer".to_string());
    }
    let mut pos = 4;
    let version = read_u8(bytes, &mut pos)?;
    if version != 1 && version != VERSION {
        return Err(format!(
            "unsupported native version {version} (this build reads 1 and {VERSION})"
        ));
    }
    let payload_len = read_u32(bytes, &mut pos)? as usize;
    let trailer = if version >= 2 { 4 } else { 0 };
    if bytes.len() - pos != payload_len + trailer {
        return Err(format!(
            "payload length mismatch: header says {payload_len}, {} bytes remain",
            bytes.len() - pos
        ));
    }
    Ok((version, pos, payload_len))
}

/// Verify the v2 CRC32 trailer against the payload. A v1 buffer
/// (no checksum) always passes.
pub(crate) fn verify_checksum(
    bytes: &[u8],
    version: u8,
    payload_start: usize,
    payload_len: usize,
) -> Result<(), String> {
    if version < 2 {
        return Ok(());
    }
    let payload_end = payload_start + payload_len;
    let mut pos = payload_end;
    let stored = read_u32(bytes, &mut pos)?;
    let actual = crc32(&bytes[payload_start..payload_end]);
    if stored != actual {
        return Err(format!(
            "integrity check failed: payload checksum {actual:#010x} does not match stored {stored:#010x}"
        ));
    }
    Ok(())
}

/// Bitwise CRC32 (IEEE 802.3, reflected, poly 0xEDB88320).
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        let mut k = 0;
        while k < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
            k += 1;
        }
    }
    !crc
}
