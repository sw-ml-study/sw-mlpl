//! TOML decoder -- the config subset into a record. Line-oriented:
//! blank and `#` comment lines are skipped, `[table]` / dotted
//! `[table.sub]` headers set the current table, and `key = value`
//! assignments parse into it (values via `toml_scalar::parse_value`,
//! which reuses the JSON value parser). Errors carry a line number.

use std::collections::BTreeMap;

use mlpl_eval_types::Value;

/// Decode the TOML config subset into a record value.
pub(crate) fn decode(text: &str, limits: &crate::decode_limits::Limits) -> Result<Value, String> {
    if text.len() > limits.max_bytes {
        return Err(format!(
            "input of {} bytes exceeds max_bytes {}",
            text.len(),
            limits.max_bytes
        ));
    }
    let mut root: BTreeMap<String, Value> = BTreeMap::new();
    let mut path: Vec<String> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |e: String| format!("{e} (line {})", i + 1);
        if line.starts_with('[') {
            path = parse_header(line).map_err(at)?;
            table_at(&mut root, &path).map_err(at)?;
        } else {
            let (k, v) = parse_kv(line, limits.max_depth).map_err(at)?;
            table_at(&mut root, &path).map_err(at)?.insert(k, v);
        }
    }
    let record = Value::Record { fields: root };
    crate::element_count::check(&record, limits.max_elements)?;
    Ok(record)
}

/// `[a.b.c]` -> `["a", "b", "c"]`; only `#` may follow the `]`.
fn parse_header(line: &str) -> Result<Vec<String>, String> {
    let close = line
        .find(']')
        .ok_or_else(|| "parse_toml: unclosed table header".to_string())?;
    let rest = line[close + 1..].trim_start();
    if !rest.is_empty() && !rest.starts_with('#') {
        return Err(format!("parse_toml: text after table header: {rest:?}"));
    }
    let segs: Vec<String> = line[1..close]
        .split('.')
        .map(|s| s.trim().to_string())
        .collect();
    for s in &segs {
        crate::toml_scalar::bare_key(s).map_err(|_| format!("parse_toml: bad table key {s:?}"))?;
    }
    Ok(segs)
}

/// `key = value` split at the first `=` (bare keys hold no `=`).
fn parse_kv(line: &str, max_depth: usize) -> Result<(String, Value), String> {
    let eq = line
        .find('=')
        .ok_or_else(|| format!("parse_toml: expected key = value, got {line:?}"))?;
    let key = line[..eq].trim();
    crate::toml_scalar::bare_key(key).map_err(|_| format!("parse_toml: bad key {key:?}"))?;
    let value = crate::toml_scalar::parse_value(line[eq + 1..].trim(), max_depth)?;
    Ok((key.to_string(), value))
}

/// Navigate (creating intermediate records) to the table named by
/// `path`, returning its field map for insertion.
fn table_at<'a>(
    root: &'a mut BTreeMap<String, Value>,
    path: &[String],
) -> Result<&'a mut BTreeMap<String, Value>, String> {
    let mut cur = root;
    for seg in path {
        let entry = cur.entry(seg.clone()).or_insert_with(|| Value::Record {
            fields: BTreeMap::new(),
        });
        cur = match entry {
            Value::Record { fields } => fields,
            _ => return Err(format!("parse_toml: {seg:?} is a value, not a table")),
        };
    }
    Ok(cur)
}
