//! Reflection over the test registry and the general annotation
//! namespace: `tests()`, `test_info(name)`, `annotations(name)`.
//! Discovery never invokes anything.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Dispatch one reflection builtin.
pub(crate) fn eval_reflect(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if name == "tests" {
        if !args.is_empty() {
            return Err(bad_arity("tests", 0, args.len()));
        }
        let items = env.tests.iter().map(|t| t.name.clone()).collect();
        return Ok(Value::StrList { items });
    }
    let key = one_string_arg(name, args, env, trace)?;
    if name == "test_info" {
        return crate::reflect_info::test_info(&key, env);
    }
    crate::reflect_info::annotations_of(&key, env, trace)
}

fn bad_arity(func: &str, expected: usize, got: usize) -> EvalError {
    EvalError::BadArity {
        func: func.into(),
        expected,
        got,
    }
}

/// The single name argument (string literal or Str expression).
fn one_string_arg(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<String, EvalError> {
    if args.len() != 1 {
        return Err(bad_arity(name, 1, args.len()));
    }
    match crate::eval::eval_expr(&args[0], env, trace)? {
        Value::Str(s) => Ok(s),
        other => Err(EvalError::Unsupported(format!(
            "{name}: the argument is a NAME string -- got {}",
            mlpl_eval_types::value_kind(&other)
        ))),
    }
}
