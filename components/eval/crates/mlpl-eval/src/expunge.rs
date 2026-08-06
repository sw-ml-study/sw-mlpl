//! `expunge(name | [names])` -- APL's quad-EX -- and the
//! `:erase` REPL command ()ERASE lineage). The result is 1 when
//! the name is FREE afterwards (already-unbound names included:
//! cleanup is idempotent) and 0 for a malformed name. Every
//! value table lets go (the `clear_binding` sweep); a `u:` name
//! also leaves the function table and the @test registry.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn eval_expunge(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "expunge".into(),
            expected: 1,
            got: args.len(),
        });
    };
    match crate::eval::eval_expr(arg, env, trace)? {
        Value::Str(name) => Ok(Value::Array(DenseArray::from_scalar(expunge_one(
            env, &name,
        )))),
        Value::StrList { items } => {
            let mask: Vec<f64> = items.iter().map(|n| expunge_one(env, n)).collect();
            Ok(Value::Array(DenseArray::from_vec(mask)))
        }
        other => Err(EvalError::Unsupported(format!(
            "expunge: takes a name string or a string list of names -- got {}",
            mlpl_eval_types::value_kind(&other)
        ))),
    }
}

/// Free one name from every table. 1 = free afterwards, 0 =
/// malformed name (nothing to free by that spelling).
fn expunge_one(env: &mut Environment, name: &str) -> f64 {
    if !well_formed(name) {
        return 0.0;
    }
    if name.starts_with("u:") {
        env.user_fns.remove(name);
        env.tests.retain(|t| t.fn_name != name);
    } else {
        env.clear_binding(name);
    }
    1.0
}

/// An identifier, optionally `u:`-prefixed -- the only spellings
/// a binding or user function can have.
pub(crate) fn well_formed(name: &str) -> bool {
    let bare = name.strip_prefix("u:").unwrap_or(name);
    let mut chars = bare.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `:erase name name...` -- the interactive form. Reports what
/// was freed and flags malformed spellings.
pub(crate) fn erase_names(env: &mut Environment, names: &[&str]) -> String {
    if names.is_empty() {
        return "usage: :erase <name> [<name> ...] -- free bindings and u: functions \
                (programmatic form: expunge(\"name\"))"
            .to_string();
    }
    let mut freed = Vec::new();
    let mut bad = Vec::new();
    for n in names {
        if expunge_one(env, n) == 1.0 {
            freed.push(*n);
        } else {
            bad.push(*n);
        }
    }
    let mut out = String::new();
    if !freed.is_empty() {
        out.push_str(&format!("erased: {}", freed.join(", ")));
    }
    if !bad.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("not a name: {}", bad.join(", ")));
    }
    out
}
