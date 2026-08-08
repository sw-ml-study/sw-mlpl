//! Opt-in semantic Result reconstruction: convert the
//! Result-shaped records that `to_json` / `to_toml` emit back
//! into `Result` values, recursively. `{ok, value}` with `ok == 1`
//! becomes `ok(value)`; `{ok, error}` with `ok == 0` becomes
//! `err(error)`; any other record recurses into its fields; and
//! non-records pass through unchanged. Enabled by the parsers'
//! `{results: 1}` option (the shape is an ordinary record, so this
//! cannot be automatic).

use std::collections::BTreeMap;

use mlpl_eval_types::Value;

/// Walk a decoded value, rebuilding Result-shaped records.
pub(crate) fn reconstruct(v: Value) -> Value {
    let Value::Record { mut fields } = v else {
        return v;
    };
    let ok_flag = match fields.get("ok") {
        Some(Value::Array(a)) if a.rank() == 0 => Some(a.data()[0]),
        _ => None,
    };
    if fields.len() == 2 {
        if ok_flag == Some(1.0) && fields.contains_key("value") {
            let payload = reconstruct(fields.remove("value").expect("value present"));
            return Value::Result {
                ok: true,
                payload: Box::new(payload),
            };
        }
        if ok_flag == Some(0.0) && fields.contains_key("error") {
            let payload = reconstruct(fields.remove("error").expect("error present"));
            return Value::Result {
                ok: false,
                payload: Box::new(payload),
            };
        }
    }
    let rebuilt: BTreeMap<String, Value> = fields
        .into_iter()
        .map(|(k, v)| (k, reconstruct(v)))
        .collect();
    Value::Record { fields: rebuilt }
}
