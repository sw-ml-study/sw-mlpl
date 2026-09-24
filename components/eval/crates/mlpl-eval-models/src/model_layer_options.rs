//! Optional trailing record argument of a layer constructor:
//! `linear(in, out, seed, {bias: 0})`, `rms_norm(dim, {eps: 0.00001})`.
//! An absent record means every option takes its default; an unknown
//! field is an error naming the ones the layer accepts.

use std::collections::BTreeMap;

use mlpl_parser::Expr;

use mlpl_eval_env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Read the numeric options `keys` (with their defaults) from the record
/// at `args[at]`, or return the defaults when there is no such argument.
pub fn numeric_options<const N: usize>(
    func: &str,
    args: &[Expr],
    at: usize,
    keys: [(&str, f64); N],
    env: &mut Environment,
) -> Result<[f64; N], EvalError> {
    let Some(arg) = args.get(at) else {
        return Ok(keys.map(|(_, default)| default));
    };
    match mlpl_eval_env::dispatch_hook::eval_or_err(arg, env, &mut None)? {
        Value::Record { fields } => from_fields(func, keys, fields),
        _ => Err(EvalError::Unsupported(format!(
            "{func}: options must be a record with fields from: {}",
            keys.map(|(k, _)| k).join(", ")
        ))),
    }
}

/// Overlay the record's fields on the defaults, rejecting unknown names.
fn from_fields<const N: usize>(
    func: &str,
    keys: [(&str, f64); N],
    fields: BTreeMap<String, Value>,
) -> Result<[f64; N], EvalError> {
    let mut out = keys.map(|(_, default)| default);
    for (name, value) in fields {
        let Some(i) = keys.iter().position(|(k, _)| *k == name) else {
            let accepted = keys.map(|(k, _)| k).join(", ");
            return Err(EvalError::Unsupported(format!(
                "{func}: unknown option '{name}' (accepted: {accepted})"
            )));
        };
        out[i] = value
            .into_array()?
            .data()
            .first()
            .copied()
            .unwrap_or(f64::NAN);
    }
    Ok(out)
}
