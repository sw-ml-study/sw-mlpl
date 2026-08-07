//! Scalar-level JSON pieces for the decoder: numbers, strings
//! (escapes + surrogate pairs), whitespace.

use mlpl_array::DenseArray;
use mlpl_eval_types::Value;

pub(crate) fn number(text: &str, bytes: &[u8], pos: &mut usize) -> Result<Value, String> {
    let start = *pos;
    if bytes.get(*pos) == Some(&b'-') {
        *pos += 1;
    }
    while matches!(bytes.get(*pos), Some(c) if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        *pos += 1;
    }
    text[start..*pos]
        .parse::<f64>()
        .map(|n| Value::Array(DenseArray::from_scalar(n)))
        .map_err(|_| format!("bad number at byte {start}"))
}

pub(crate) fn string(text: &str, bytes: &[u8], pos: &mut usize) -> Result<String, String> {
    if bytes.get(*pos) != Some(&b'"') {
        return Err(format!("expected a string at byte {}", *pos));
    }
    *pos += 1;
    let mut out = String::new();
    loop {
        match bytes.get(*pos) {
            None => return Err("unterminated string".to_string()),
            Some(b'"') => {
                *pos += 1;
                return Ok(out);
            }
            Some(b'\\') => {
                *pos += 1;
                escape(text, bytes, pos, &mut out)?;
            }
            Some(_) => {
                let ch = text[*pos..].chars().next().ok_or("bad utf-8")?;
                out.push(ch);
                *pos += ch.len_utf8();
            }
        }
    }
}

fn escape(text: &str, bytes: &[u8], pos: &mut usize, out: &mut String) -> Result<(), String> {
    let c = *bytes.get(*pos).ok_or("unterminated escape")?;
    *pos += 1;
    match c {
        b'"' => out.push('"'),
        b'\\' => out.push('\\'),
        b'/' => out.push('/'),
        b'n' => out.push('\n'),
        b't' => out.push('\t'),
        b'r' => out.push('\r'),
        b'b' => out.push('\u{8}'),
        b'f' => out.push('\u{c}'),
        b'u' => {
            let hi = hex4(text, pos)?;
            let cp = if (0xD800..0xDC00).contains(&hi) {
                if bytes.get(*pos) != Some(&b'\\') || bytes.get(*pos + 1) != Some(&b'u') {
                    return Err("lone surrogate".to_string());
                }
                *pos += 2;
                let lo = hex4(text, pos)?;
                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
            } else {
                hi
            };
            out.push(char::from_u32(cp).ok_or("bad \\u escape")?);
        }
        other => return Err(format!("unknown escape \\{}", other as char)),
    }
    Ok(())
}

fn hex4(text: &str, pos: &mut usize) -> Result<u32, String> {
    let hex = text.get(*pos..*pos + 4).ok_or("truncated \\u escape")?;
    *pos += 4;
    u32::from_str_radix(hex, 16).map_err(|_| format!("bad hex `{hex}`"))
}

pub(crate) fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while matches!(bytes.get(*pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        *pos += 1;
    }
}
