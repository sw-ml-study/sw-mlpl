//! Evaluation helpers for binops, function calls, and array literals.

use mlpl_array::{ArrayError, DenseArray, Shape};
use mlpl_core::LabeledShape;
use mlpl_parser::{BinOpKind, Expr};
use mlpl_trace::{Trace, TraceValue};

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;
use crate::value::Value;

pub(crate) fn eval_binop(
    op: &BinOpKind,
    lhs: &Expr,
    rhs: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let l = eval_expr(lhs, env, trace)?.into_array()?;
    let r = eval_expr(rhs, env, trace)?.into_array()?;
    let name: &str = match op {
        BinOpKind::Add => "add",
        BinOpKind::Sub => "sub",
        BinOpKind::Mul => "mul",
        BinOpKind::Div => "div",
    };
    let inputs = vec![TraceValue::from_array(&l), TraceValue::from_array(&r)];
    // Saga 14 step 004/005: route the binop through the active
    // device. `dispatched_call` falls back to the CPU path for
    // everything outside a `device("mlx")` block and for ops that
    // `mlpl-mlx` does not implement. Shape/label mismatches get
    // lifted into the Saga 11.5 `EvalError::ShapeMismatch` shape.
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

/// Snapshot an array's dims + label list as a `LabeledShape`. Unlabeled
/// arrays get `None` at every axis. Saga 11.5 Phase 4: used when
/// lifting `ArrayError::ShapeMismatch` / `LabelMismatch` into
/// `EvalError::ShapeMismatch` at op call sites.
pub(crate) fn labeled_shape_of(a: &DenseArray) -> LabeledShape {
    let labels = a
        .labels()
        .map_or_else(|| vec![None; a.rank()], <[_]>::to_vec);
    LabeledShape::new(a.shape().dims().to_vec(), labels)
}

pub(crate) fn eval_fncall(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let evaluated: Vec<DenseArray> = args
        .iter()
        .map(|a| eval_expr(a, env, trace).and_then(Value::into_array))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs: Vec<TraceValue> = evaluated.iter().map(TraceValue::from_array).collect();
    // Saga 14 steps 004/005: one routing helper decides CPU vs
    // MLX, so a Model DSL `apply(...)` forward and a raw user
    // `foo(x)` call see the same dispatch rules.
    let result = crate::device::dispatched_call(env, name, evaluated)?;
    Ok(("fncall", inputs, result))
}

pub(crate) fn eval_svg(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<String, EvalError> {
    if args.len() != 2 && args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "svg".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let data = eval_expr(&args[0], env, trace)?.into_array()?;
    let type_name = match eval_expr(&args[1], env, trace)? {
        Value::Str(s) => s,
        Value::Array(_) | Value::Model(_) | Value::Tokenizer(_) => {
            return Err(EvalError::ExpectedString);
        }
    };
    let aux = if args.len() == 3 {
        Some(eval_expr(&args[2], env, trace)?.into_array()?)
    } else {
        None
    };
    Ok(mlpl_viz::render_with_aux(&data, &type_name, aux.as_ref())?)
}

/// Dispatch high-level analysis helpers from `mlpl_viz`. Returns
/// `None` if `name` is not a known helper.
pub(crate) fn eval_analysis_helper(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<String, EvalError>> {
    if !matches!(
        name,
        "hist" | "scatter_labeled" | "loss_curve" | "confusion_matrix" | "boundary_2d"
    ) {
        return None;
    }
    let evaluated: Result<Vec<DenseArray>, EvalError> = args
        .iter()
        .map(|a| eval_expr(a, env, trace).and_then(Value::into_array))
        .collect();
    Some(match evaluated {
        Ok(arrs) => call_analysis(name, &arrs),
        Err(e) => Err(e),
    })
}

fn call_analysis(name: &str, a: &[DenseArray]) -> Result<String, EvalError> {
    let bad_arity = |expected: usize| EvalError::BadArity {
        func: name.into(),
        expected,
        got: a.len(),
    };
    Ok(match name {
        "hist" => {
            if a.len() != 2 {
                return Err(bad_arity(2));
            }
            if a[1].rank() != 0 {
                return Err(EvalError::ExpectedString);
            }
            mlpl_viz::analysis_hist(&a[0], a[1].data()[0] as usize)?
        }
        "scatter_labeled" => {
            if a.len() != 2 {
                return Err(bad_arity(2));
            }
            mlpl_viz::analysis_scatter_labeled(&a[0], &a[1])?
        }
        "loss_curve" => {
            if a.len() != 1 {
                return Err(bad_arity(1));
            }
            mlpl_viz::analysis_loss_curve(&a[0])?
        }
        "confusion_matrix" => {
            if a.len() != 2 {
                return Err(bad_arity(2));
            }
            mlpl_viz::analysis_confusion_matrix(&a[0], &a[1])?
        }
        "boundary_2d" => {
            if a.len() != 4 {
                return Err(bad_arity(4));
            }
            mlpl_viz::analysis_boundary_2d(&a[0], &a[1], &a[2], &a[3])?
        }
        _ => unreachable!(),
    })
}

pub(crate) fn eval_array_lit(
    elems: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if elems.is_empty() {
        return Ok(DenseArray::from_vec(vec![]));
    }
    let evaluated: Vec<DenseArray> = elems
        .iter()
        .map(|e| eval_expr(e, env, trace).and_then(Value::into_array))
        .collect::<Result<Vec<_>, _>>()?;
    if evaluated.iter().all(|a| a.rank() == 0) {
        let data: Vec<f64> = evaluated.iter().map(|a| a.data()[0]).collect();
        return Ok(DenseArray::from_vec(data));
    }
    let inner_shape = evaluated[0].shape().clone();
    let rows = evaluated.len();
    let mut data = Vec::with_capacity(rows * inner_shape.elem_count());
    for arr in &evaluated {
        data.extend_from_slice(arr.data());
    }
    let mut dims = vec![rows];
    dims.extend_from_slice(inner_shape.dims());
    Ok(DenseArray::new(Shape::new(dims), data)?)
}
