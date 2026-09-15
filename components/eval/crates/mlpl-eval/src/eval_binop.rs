//! `Expr::BinOp` evaluation. `+` on two strings concatenates
//! (demo-coding-agent CA1); every other operand combination goes through the
//! array/tensor arithmetic path. Extracted from `eval.rs` so `eval_expr` and
//! that file stay under their length budgets; the array path keeps the Saga
//! 11.5 shape/label-mismatch lifting and the trace event a binop emits.

use mlpl_array::{ArrayError, DenseArray};
use mlpl_parser::{BinOpKind, Expr};
use mlpl_trace::{Trace, TraceEvent, TraceValue};

use crate::env::Environment;
use crate::eval::eval_expr;
use crate::eval_ops::labeled_shape_of;
use mlpl_eval_types::{EvalError, Value};

/// Evaluate `lhs <op> rhs`. Operands are evaluated exactly once (so a
/// side-effecting operand is safe); two strings under `+` concatenate,
/// otherwise the operands are coerced to arrays and the op runs on the tape /
/// device with a trace event recorded.
pub(crate) fn eval_binop(
    expr: &Expr,
    op: &BinOpKind,
    lhs: &Expr,
    rhs: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let lv = eval_expr(lhs, env, trace)?;
    let rv = eval_expr(rhs, env, trace)?;
    if let (BinOpKind::Add, Value::Str(a), Value::Str(b)) = (op, &lv, &rv) {
        return Ok(Value::Str(format!("{a}{b}")));
    }
    let (name, inputs, result) = binop_arrays(op, lv, rv, env)?;
    if let Some(t) = trace.as_mut() {
        let seq = t.events().len() as u64;
        let (input_types, output_type) = crate::auto_tag::for_trace_event(expr, env);
        t.push(TraceEvent {
            seq,
            op: name.into(),
            span: expr.span(),
            inputs,
            output: TraceValue::from_array(&result),
            input_types,
            output_type,
        });
    }
    Ok(Value::Array(result))
}

/// The array/tensor arithmetic path: coerce both operands to arrays and run the
/// op through the active device, lifting a shape/label mismatch into the richer
/// `EvalError::ShapeMismatch`.
fn binop_arrays(
    op: &BinOpKind,
    l_val: Value,
    r_val: Value,
    env: &mut Environment,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let l = l_val.into_array()?;
    let r = r_val.into_array()?;
    let name: &str = match op {
        BinOpKind::Add => "add",
        BinOpKind::Sub => "sub",
        BinOpKind::Mul => "mul",
        BinOpKind::Div => "div",
        BinOpKind::Lt => "lt",
        BinOpKind::Gt => "gt",
        BinOpKind::Le => "le",
        BinOpKind::Ge => "ge",
        BinOpKind::Eq => "eq",
        BinOpKind::Ne => "ne",
    };
    let inputs = vec![TraceValue::from_array(&l), TraceValue::from_array(&r)];
    let result = match crate::device::dispatched_call(env, name, vec![l.clone(), r.clone()]) {
        Ok(a) => a,
        Err(EvalError::ArrayError(
            ArrayError::ShapeMismatch { .. } | ArrayError::LabelMismatch { .. },
        )) => {
            return Err(EvalError::ShapeMismatch {
                op: name.into(),
                expected: labeled_shape_of(&l),
                actual: labeled_shape_of(&r),
            });
        }
        Err(e) => return Err(e),
    };
    Ok((name, inputs, result))
}
