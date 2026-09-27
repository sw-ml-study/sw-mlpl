//! Values to text: `to_string(x)` (a scalar's round-trip decimal) and
//! `format(template, args...)` -> a string, with Python `str.format`
//! replacement fields (`{}`, `{0}`, `{:>8}`, `{:4d}`, `{:.4f}`, `{:e}`,
//! `{:,}`, `{:.1%}`, ...). The formatting itself is the pure
//! `mlpl-format` crate; this module evaluates the arguments: a scalar is a
//! number, a string is text, and any other value formats as its display
//! text (so `{}` works on anything, a numeric spec on it errors).

use mlpl_format::FmtArg;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn eval_format(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let Some((template, rest)) = args.split_first() else {
        return Err(EvalError::BadArity {
            func: "format".into(),
            expected: 1,
            got: 0,
        });
    };
    let Value::Str(template) = eval_expr(template, env, trace)? else {
        return Err(EvalError::Unsupported(
            "format: the first argument must be the template string".into(),
        ));
    };
    let values = rest
        .iter()
        .map(|a| eval_expr(a, env, trace).map(format_arg))
        .collect::<Result<Vec<_>, _>>()?;
    mlpl_format::format(&template, &values)
        .map(Value::Str)
        .map_err(EvalError::Unsupported)
}

/// `to_string(x)` -> the shortest round-trip decimal of a scalar
/// number, the honest inverse of `to_number`: integral values print
/// bare (`to_string(8 / 2)` is `"4"`, not `"4.0"`) using the same
/// formatting `to_json` gives a scalar, so `to_number(to_string(x))`
/// recovers `x` for every finite `f64`. A non-scalar / non-number is
/// an error; `format` is the spec-driven alternative.
pub(crate) fn eval_to_string(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [x] = args else {
        return Err(EvalError::BadArity {
            func: "to_string".into(),
            expected: 1,
            got: args.len(),
        });
    };
    match eval_expr(x, env, trace)? {
        Value::Array(a) if a.rank() == 0 => {
            let mut s = String::new();
            crate::json_encode::push_number(&mut s, a.data()[0]);
            Ok(Value::Str(s))
        }
        _ => Err(EvalError::Unsupported(
            "to_string: expected a scalar number".into(),
        )),
    }
}

/// `write(...)`: `print` without the newline -- the rendered pieces
/// concatenated, written to stdout and flushed (so `\r` lines update in
/// place). The text written is the call's value.
pub(crate) fn emit_write(rendered: &[String]) -> Value {
    let text = rendered.concat();
    print!("{text}");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    Value::Str(text)
}

/// A scalar is a number; a string is itself; anything else is its display.
fn format_arg(v: Value) -> FmtArg {
    match v {
        Value::Array(a) if a.rank() == 0 => FmtArg::Num(a.data()[0]),
        Value::Str(s) => FmtArg::Text(s),
        other => FmtArg::Text(other.to_string()),
    }
}
