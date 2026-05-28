//! Higher-order `reduce(:op, x[, axis])` dispatch.
//!
//! The first argument is a `Value::BuiltinRef` naming a
//! binary operation: `:add`, `:mul` (or `:+` / `:*`),
//! `:min`, `:max`, `:and`, `:or`. The fixed-name builtins
//! `reduce_add` and `reduce_mul` continue to exist as
//! direct shorthands.
//!
//! Why `:op` instead of a string or bare-name reference?
//! MLPL has no first-class functions in v0.19; the colon-
//! prefixed BuiltinRef occupies a separate syntactic
//! namespace from variables, so a user's `add = 42` cannot
//! shadow `:add`. The same surface stays valid when
//! first-class functions land -- `:foo` will lift to a
//! `Value::Function::Builtin("foo")` cleanly.

use mlpl_array::DenseArray;
use mlpl_array_ops_reduce::prelude::*;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;
use crate::value::Value;

type BinOp = fn(f64, f64) -> f64;

fn add_op(a: f64, b: f64) -> f64 {
    a + b
}

fn mul_op(a: f64, b: f64) -> f64 {
    a * b
}

fn min_op(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

fn max_op(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

fn and_op(a: f64, b: f64) -> f64 {
    if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }
}

fn or_op(a: f64, b: f64) -> f64 {
    if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }
}

fn dispatch(op_name: &str) -> Option<(f64, BinOp)> {
    match op_name {
        "add" | "+" => Some((0.0, add_op)),
        "mul" | "*" => Some((1.0, mul_op)),
        "min" => Some((f64::INFINITY, min_op)),
        "max" => Some((f64::NEG_INFINITY, max_op)),
        "and" => Some((1.0, and_op)),
        "or" => Some((0.0, or_op)),
        _ => None,
    }
}

pub(crate) fn eval_reduce(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(EvalError::BadArity {
            func: "reduce".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let op_name = match eval_expr(&args[0], env, trace)? {
        Value::BuiltinRef { name } => name,
        _ => {
            return Err(EvalError::Unsupported(
                "reduce: first argument must be a builtin reference like :add, :max, :+, :* (use the colon-prefixed form)".into(),
            ));
        }
    };
    let (identity, op) = dispatch(&op_name).ok_or_else(|| {
        EvalError::Unsupported(format!(
            "reduce: unknown op ':{op_name}' (curated set: :add/:+, :mul/:*, :min, :max, :and, :or)"
        ))
    })?;
    let arr = eval_expr(&args[1], env, trace)?.into_array()?;
    let result = if args.len() == 3 {
        let axis_arr = eval_expr(&args[2], env, trace)?.into_array()?;
        if axis_arr.rank() != 0 {
            return Err(EvalError::Unsupported(format!(
                "reduce: axis must be a scalar, got rank {}",
                axis_arr.rank()
            )));
        }
        let axis = axis_arr.data()[0] as usize;
        arr.reduce_axis(axis, identity, op)?
    } else {
        DenseArray::from_scalar(arr.data().iter().copied().fold(identity, op))
    };
    Ok(Value::Array(result))
}
