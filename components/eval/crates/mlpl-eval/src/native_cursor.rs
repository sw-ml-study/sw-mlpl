//! Bounds-checked byte-cursor reads for the native binary decoder.
//! Each read advances `*pos` and errors (rather than panicking) on
//! truncated input, so a corrupt/short buffer is an err Result.

/// Read one byte.
pub(crate) fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8, String> {
    let b = *bytes
        .get(*pos)
        .ok_or_else(|| format!("truncated: expected a byte at {}", *pos))?;
    *pos += 1;
    Ok(b)
}

/// Read a little-endian u32.
pub(crate) fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, String> {
    let end = *pos + 4;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| format!("truncated: expected 4 bytes at {}", *pos))?;
    *pos = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

/// Read a little-endian f64.
pub(crate) fn read_f64(bytes: &[u8], pos: &mut usize) -> Result<f64, String> {
    let end = *pos + 8;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| format!("truncated: expected 8 bytes at {}", *pos))?;
    *pos = end;
    Ok(f64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

/// Read a length-prefixed UTF-8 string (u32 LE len + bytes).
pub(crate) fn read_str(bytes: &[u8], pos: &mut usize) -> Result<String, String> {
    let len = read_u32(bytes, pos)? as usize;
    let end = *pos + len;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| format!("truncated: expected {len} string bytes at {}", *pos))?;
    *pos = end;
    String::from_utf8(slice.to_vec()).map_err(|e| format!("invalid UTF-8 string: {e}"))
}
