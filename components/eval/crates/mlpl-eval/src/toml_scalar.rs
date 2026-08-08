//! Scalar-level helpers shared by the TOML codec: bare-key
//! validation, number formatting (bare integer or float, with a
//! non-finite guard), and array encoding. String escaping reuses
//! the JSON encoder's `push_str_json` (TOML basic strings share
//! JSON's escape set).

use mlpl_array::DenseArray;
use mlpl_eval_types::Value;

/// Parse one TOML value (the text after `=`, or an array element)
/// by delegating to the JSON value parser -- TOML basic strings,
/// numbers, booleans (`true`/`false` -> 1/0), and arrays are a
/// subset of JSON values. A trailing `# comment` is allowed.
pub(crate) fn parse_value(s: &str, max_depth: usize) -> Result<Value, String> {
    if s.is_empty() {
        return Err("parse_toml: empty value".to_string());
    }
    let bytes = s.as_bytes();
    let mut pos = 0;
    let v = crate::json_decode::value(s, bytes, &mut pos, max_depth)?;
    crate::json_scalar::skip_ws(bytes, &mut pos);
    if pos < bytes.len() && bytes[pos] != b'#' {
        return Err(format!(
            "parse_toml: unexpected text after value: {:?}",
            &s[pos..]
        ));
    }
    Ok(v)
}

/// A bare TOML key is non-empty ASCII `[A-Za-z0-9_-]`. Quoted and
/// dotted keys are out of this subset -- a loud error names the
/// offender.
pub(crate) fn bare_key(k: &str) -> Result<&str, String> {
    let bare = !k.is_empty()
        && k.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if bare {
        Ok(k)
    } else {
        Err(format!(
            "to_toml: key {k:?} is not a bare TOML key (only A-Za-z0-9_-)"
        ))
    }
}

/// Bare integer for integral values, else a float; a non-finite
/// number (NaN / +-Inf) has no TOML form and is an error.
pub(crate) fn push_number(n: f64, out: &mut String) -> Result<(), String> {
    if !n.is_finite() {
        return Err(format!(
            "to_toml: cannot serialize the non-finite number {n} (TOML has no NaN or infinity)"
        ));
    }
    if n.fract() == 0.0 && n.abs() < 1e15 {
        out.push_str(&format!("{}", n as i64));
    } else {
        out.push_str(&format!("{n}"));
    }
    Ok(())
}

/// rank-0 -> a bare number; rank-1 -> `[a, b, c]`. Higher rank
/// has no TOML value form here and is an error.
pub(crate) fn encode_array(a: &DenseArray, out: &mut String) -> Result<(), String> {
    let dims = a.shape().dims();
    if dims.len() > 1 {
        return Err(format!(
            "to_toml: cannot represent a rank-{} array as a TOML value",
            dims.len()
        ));
    }
    if dims.is_empty() {
        return push_number(a.data()[0], out);
    }
    out.push('[');
    for (i, &n) in a.data().iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_number(n, out)?;
    }
    out.push(']');
    Ok(())
}
