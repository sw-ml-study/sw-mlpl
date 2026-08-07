//! Recursive-descent JSON decoder for `parse_json`: byte cursor
//! over UTF-8 with exact string reconstruction (escape and
//! \uXXXX handling, surrogate pairs included).

use std::collections::BTreeMap;

use mlpl_array::DenseArray;
use mlpl_eval_types::Value;

pub(crate) fn decode(text: &str) -> Result<Value, String> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    crate::json_scalar::skip_ws(bytes, &mut pos);
    let v = value(text, bytes, &mut pos)?;
    crate::json_scalar::skip_ws(bytes, &mut pos);
    if pos != bytes.len() {
        return Err(format!("trailing input at byte {pos}"));
    }
    Ok(v)
}

pub(crate) fn value(text: &str, bytes: &[u8], pos: &mut usize) -> Result<Value, String> {
    match bytes.get(*pos) {
        Some(b'{') => object(text, bytes, pos),
        Some(b'[') => array(text, bytes, pos),
        Some(b'"') => Ok(Value::Str(crate::json_scalar::string(text, bytes, pos)?)),
        Some(b't') if text[*pos..].starts_with("true") => {
            *pos += 4;
            Ok(Value::Array(DenseArray::from_scalar(1.0)))
        }
        Some(b'f') if text[*pos..].starts_with("false") => {
            *pos += 5;
            Ok(Value::Array(DenseArray::from_scalar(0.0)))
        }
        Some(b'n') if text[*pos..].starts_with("null") => {
            *pos += 4;
            Ok(Value::Array(DenseArray::from_vec(Vec::new())))
        }
        Some(c) if c.is_ascii_digit() || *c == b'-' => crate::json_scalar::number(text, bytes, pos),
        _ => Err(format!("expected a JSON value at byte {}", *pos)),
    }
}

fn object(text: &str, bytes: &[u8], pos: &mut usize) -> Result<Value, String> {
    *pos += 1;
    let mut fields = BTreeMap::new();
    crate::json_scalar::skip_ws(bytes, pos);
    if bytes.get(*pos) == Some(&b'}') {
        *pos += 1;
        return Ok(Value::Record { fields });
    }
    loop {
        crate::json_scalar::skip_ws(bytes, pos);
        let key = crate::json_scalar::string(text, bytes, pos)?;
        crate::json_scalar::skip_ws(bytes, pos);
        if bytes.get(*pos) != Some(&b':') {
            return Err(format!("expected ':' at byte {}", *pos));
        }
        *pos += 1;
        crate::json_scalar::skip_ws(bytes, pos);
        fields.insert(key, value(text, bytes, pos)?);
        crate::json_scalar::skip_ws(bytes, pos);
        match bytes.get(*pos) {
            Some(b',') => *pos += 1,
            Some(b'}') => {
                *pos += 1;
                return Ok(Value::Record { fields });
            }
            _ => return Err(format!("expected ',' or '}}' at byte {}", *pos)),
        }
    }
}

/// Homogeneous arrays only: all numbers (vector) or all
/// strings (string list); an empty array is the empty vector.
fn array(text: &str, bytes: &[u8], pos: &mut usize) -> Result<Value, String> {
    *pos += 1;
    let mut nums: Vec<f64> = Vec::new();
    let mut strs: Vec<String> = Vec::new();
    crate::json_scalar::skip_ws(bytes, pos);
    if bytes.get(*pos) == Some(&b']') {
        *pos += 1;
        return Ok(Value::Array(DenseArray::from_vec(Vec::new())));
    }
    loop {
        crate::json_scalar::skip_ws(bytes, pos);
        match value(text, bytes, pos)? {
            Value::Array(a) if a.rank() == 0 && strs.is_empty() => nums.push(a.data()[0]),
            Value::Str(s) if nums.is_empty() => strs.push(s),
            _ => return Err(format!("mixed or nested array near byte {}", *pos)),
        }
        crate::json_scalar::skip_ws(bytes, pos);
        match bytes.get(*pos) {
            Some(b',') => *pos += 1,
            Some(b']') => {
                *pos += 1;
                return Ok(if strs.is_empty() {
                    Value::Array(DenseArray::from_vec(nums))
                } else {
                    Value::StrList { items: strs }
                });
            }
            _ => return Err(format!("expected ',' or ']' at byte {}", *pos)),
        }
    }
}
