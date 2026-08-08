//! Shared decode limits for the parsers (`parse_json`,
//! `parse_toml`): a max nesting depth (guarding the recursive-
//! descent decoder against stack overflow on adversarial input)
//! and a max input byte size. Depth is always enforced with a
//! sensible default; both are overridable via an optional options
//! record passed as the codec's second argument.

use std::collections::BTreeMap;

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Default max nesting depth (matches serde_json's default).
pub(crate) const DEFAULT_MAX_DEPTH: usize = 128;

pub(crate) struct Limits {
    pub(crate) max_depth: usize,
    pub(crate) max_bytes: usize,
}

impl Limits {
    fn defaults() -> Self {
        Limits {
            max_depth: DEFAULT_MAX_DEPTH,
            max_bytes: usize::MAX,
        }
    }
}

/// Evaluate a parser's `(text[, options])` arguments into the
/// input string and its resolved limits. A missing options arg
/// keeps the defaults; a malformed call (bad arity, non-string
/// text, non-record or bad-field options) is a hard error --
/// misuse of the call, distinct from bad input data (an err
/// Result).
pub(crate) fn text_and_limits(
    who: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(String, Limits), EvalError> {
    if args.is_empty() || args.len() > 2 {
        return Err(EvalError::BadArity {
            func: who.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let Value::Str(text) = crate::eval::eval_expr(&args[0], env, trace)? else {
        return Err(EvalError::Unsupported(format!(
            "{who}: the first argument is text (a string)"
        )));
    };
    let opt = match args.get(1) {
        Some(a) => Some(crate::eval::eval_expr(a, env, trace)?),
        None => None,
    };
    Ok((text, from_option(who, opt.as_ref())?))
}

fn from_option(who: &str, opt: Option<&Value>) -> Result<Limits, EvalError> {
    let mut limits = Limits::defaults();
    let Some(v) = opt else {
        return Ok(limits);
    };
    let Value::Record { fields } = v else {
        return Err(EvalError::Unsupported(format!(
            "{who}: the options argument must be a record"
        )));
    };
    if let Some(d) = usize_field(who, fields, "max_depth")? {
        limits.max_depth = d;
    }
    if let Some(b) = usize_field(who, fields, "max_bytes")? {
        limits.max_bytes = b;
    }
    Ok(limits)
}

fn usize_field(
    who: &str,
    fields: &BTreeMap<String, Value>,
    key: &str,
) -> Result<Option<usize>, EvalError> {
    match fields.get(key) {
        None => Ok(None),
        Some(Value::Array(a))
            if a.rank() == 0 && a.data()[0] >= 0.0 && a.data()[0].fract() == 0.0 =>
        {
            Ok(Some(a.data()[0] as usize))
        }
        Some(_) => Err(EvalError::Unsupported(format!(
            "{who}: {key} must be a non-negative integer"
        ))),
    }
}
