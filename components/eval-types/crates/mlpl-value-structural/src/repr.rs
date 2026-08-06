//! Bounded deterministic rendering for diagnostics. One runtime
//! version renders one value one way; large values truncate with
//! an explicit elision marker. NOT a serialization format.

use crate::fmt_util::{join, write_str};
use mlpl_eval_types::Value;

const MAX_ELEMS: usize = 16;
const MAX_CHARS: usize = 400;

/// Render `v` for expected/actual diagnostics.
#[must_use]
pub fn value_repr(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v);
    if out.len() > MAX_CHARS {
        let cut = out
            .char_indices()
            .take_while(|(i, _)| *i <= MAX_CHARS)
            .last()
            .map_or(0, |(i, _)| i);
        out.truncate(cut);
        out.push_str("...");
    }
    out
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Array(a) => write_array(out, a),
        Value::Str(s) => write_str(out, s),
        Value::StrList { .. } | Value::Record { .. } => write_container(out, v),
        Value::Result { ok, payload } => {
            out.push_str(if *ok { "ok(" } else { "err(" });
            write_value(out, payload);
            out.push(')');
        }
        Value::Model(spec) => {
            out.push_str("model(params: ");
            let ps = spec.params();
            join(out, ps.len().min(MAX_ELEMS), |o, i| o.push_str(&ps[i]));
            out.push(')');
        }
        Value::Tokenizer(_) => out.push_str("tokenizer(...)"),
        Value::BuiltinRef { name } => {
            out.push(':');
            out.push_str(name);
        }
        Value::DeviceTensor { .. } => out.push_str("device-tensor(...)"),
    }
}

/// `StrList` and `Record` bodies, split out for the LOC budget.
fn write_container(out: &mut String, v: &Value) {
    match v {
        Value::StrList { items } => {
            out.push('[');
            join(out, items.len(), |o, i| write_str(o, &items[i]));
            out.push(']');
        }
        Value::Record { fields } => {
            out.push('{');
            let fs: Vec<_> = fields.iter().collect();
            join(out, fs.len(), |o, i| {
                o.push_str(fs[i].0);
                o.push_str(": ");
                write_value(o, fs[i].1);
            });
            out.push('}');
        }
        _ => {}
    }
}

fn write_array(out: &mut String, a: &mlpl_array::DenseArray) {
    out.push_str("array[");
    join(out, a.shape().dims().len(), |o, i| {
        o.push_str(&a.shape().dims()[i].to_string());
    });
    out.push(']');
    if let Some(labels) = a.labels() {
        out.push('{');
        join(out, labels.len(), |o, i| {
            o.push_str(labels[i].as_deref().unwrap_or("_"));
        });
        out.push('}');
    }
    out.push_str(" [");
    let n = a.data().len();
    join(out, n.min(MAX_ELEMS), |o, i| {
        o.push_str(&a.data()[i].to_string());
    });
    if n > MAX_ELEMS {
        use std::fmt::Write as _;
        let _ = write!(out, " ... ({n} values)");
    }
    out.push(']');
}
