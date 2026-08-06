//! AST-walking evaluator: the central `eval_expr` dispatcher.
//! Big-shape constructor helpers (`eval_tensor_ctor`,
//! `eval_repeat`, `eval_train`) live in `eval_blocks.rs`; the
//! program-level entry points (`eval_program*` and the
//! `run_program` loop) live in `eval_program.rs`.

use crate::env_api::*;
use mlpl_array::DenseArray;
use mlpl_array_ops_element::prelude::*;
use mlpl_parser::{Expr, TensorCtorKind};
use mlpl_trace::{Trace, TraceEvent, TraceValue};

use crate::env::Environment;
use crate::eval_ops::{eval_binop, eval_fncall, eval_svg, flatten_evaluated_arrays};
use mlpl_eval_types::EvalError;
use mlpl_eval_types::{Value, value_kind};

pub(crate) fn eval_expr(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if let Expr::StrLit(s, _) = expr {
        return Ok(Value::Str(s.clone()));
    }
    if let Expr::Include(path, _) = expr {
        return Err(EvalError::Unsupported(format!(
            "include \"{path}\": include is a script-mode construct -- run the file \
             under `mlpl-repl -f` (optionally with --source-dir); this surface has \
             no source provider"
        )));
    }
    if let Expr::BuiltinRef(name, _) = expr {
        if name.starts_with("u:") {
            return Ok(Value::UserFnRef { name: name.clone() });
        }
        return Ok(Value::BuiltinRef { name: name.clone() });
    }
    if let Expr::RecordLit { fields, .. } = expr {
        let mut map = std::collections::BTreeMap::new();
        for (name, value_expr) in fields {
            let v = eval_expr(value_expr, env, trace)?;
            map.insert(name.clone(), v);
        }
        return Ok(Value::Record { fields: map });
    }
    // Saga 29 step 002: `[...]` array-literal dispatch.
    // Evaluate each element first, then decide based on the
    // resulting kinds:
    //   - all `Value::Str`  -> `Value::StrList`
    //   - all `Value::Array` (numeric) -> existing numeric
    //     flattening, with an `array_lit` trace event so the
    //     prior trace-event shape is preserved
    //   - empty `[]` keeps the back-compat numeric path
    //     (empty `DenseArray`)
    //   - mixed -> `MixedArrayLitElements`
    if let Expr::ArrayLit(elems, _) = expr {
        let arr_opt: Option<DenseArray> = if elems.is_empty() {
            Some(DenseArray::from_vec(vec![]))
        } else {
            let mut evaluated: Vec<Value> = Vec::with_capacity(elems.len());
            for e in elems {
                evaluated.push(eval_expr(e, env, trace)?);
            }
            if evaluated.iter().all(|v| matches!(v, Value::Str(_))) {
                let items: Vec<String> = evaluated
                    .into_iter()
                    .map(|v| match v {
                        Value::Str(s) => s,
                        _ => unreachable!(),
                    })
                    .collect();
                return Ok(Value::StrList { items });
            }
            if evaluated.iter().all(|v| matches!(v, Value::Array(_))) {
                let arrays: Vec<DenseArray> = evaluated
                    .into_iter()
                    .map(|v| match v {
                        Value::Array(a) => a,
                        _ => unreachable!(),
                    })
                    .collect();
                Some(flatten_evaluated_arrays(arrays)?)
            } else {
                let kinds: Vec<&'static str> = evaluated.iter().map(value_kind).collect();
                return Err(EvalError::MixedArrayLitElements { kinds });
            }
        };
        let arr = arr_opt.unwrap();
        if let Some(t) = trace.as_mut() {
            let seq = t.events().len() as u64;
            t.push(TraceEvent {
                seq,
                op: "array_lit".into(),
                span: expr.span(),
                inputs: vec![],
                output: TraceValue::from_array(&arr),
                input_types: vec![],
                output_type: None,
            });
        }
        return Ok(Value::Array(arr));
    }
    // Saga 31 step 005 refactor: scripting-cluster and Result
    // FnCall intercepts moved to a sibling module per
    // docs/code_metrics.md (split eval.rs by responsibility).
    if let Some(result) = crate::eval_intercepts::try_intercept(expr, env, trace) {
        return result;
    }
    if let Expr::FieldAccess {
        receiver, field, ..
    } = expr
    {
        let recv = eval_expr(receiver, env, trace)?;
        return match recv {
            Value::Record { fields } => {
                fields
                    .get(field)
                    .cloned()
                    .ok_or_else(|| EvalError::FieldNotFound {
                        requested: field.clone(),
                        available: fields.keys().cloned().collect(),
                    })
            }
            other => Err(EvalError::FieldOnNonRecord {
                receiver_kind: value_kind(&other),
                field: field.clone(),
            }),
        };
    }
    if let Expr::Ident(name, _) = expr
        && let Some(s) = env.get_string(name)
    {
        return Ok(Value::Str(s.clone()));
    }
    if let Expr::Ident(name, _) = expr
        && let Some(target) = env.get_builtin_ref(name)
    {
        if target.starts_with("u:") {
            return Ok(Value::UserFnRef {
                name: target.clone(),
            });
        }
        return Ok(Value::BuiltinRef {
            name: target.clone(),
        });
    }
    if let Expr::Ident(name, _) = expr
        && let Some(v) = env.get_device_tensor(name)
    {
        return Ok(v.clone());
    }
    if let Expr::Ident(name, _) = expr
        && let Some(fields) = env.get_record(name)
    {
        return Ok(Value::Record {
            fields: fields.clone(),
        });
    }
    if let Expr::Ident(name, _) = expr
        && let Some(items) = env.get_string_list(name)
    {
        return Ok(Value::StrList {
            items: items.clone(),
        });
    }
    if let Expr::Ident(name, _) = expr
        && let Some((ok, payload)) = env.get_result(name)
    {
        return Ok(Value::Result {
            ok: *ok,
            payload: Box::new(payload.clone()),
        });
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "svg"
    {
        return eval_svg(args, env, trace).map(Value::Str);
    }
    if let Expr::FnCall { name, args, span } = expr
        && name == "grad"
    {
        let result = crate::grad::eval_grad(args, env)?;
        if let Some(t) = trace.as_mut() {
            let seq = t.events().len() as u64;
            t.push(TraceEvent {
                seq,
                op: "grad".into(),
                span: *span,
                inputs: vec![],
                output: TraceValue::from_array(&result),
                input_types: vec![],
                output_type: None,
            });
        }
        return Ok(Value::Array(result));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name.contains(':')
    {
        return crate::eval_user_fn::call_user_fn(name, args, env, trace);
    }
    if let Some(r) = crate::eval_fncalls::try_dispatch(expr, env, trace) {
        return r;
    }
    if let Expr::Device { target, body, .. } = expr {
        return crate::device::eval_device(target, body, env, trace);
    }
    if let Expr::If {
        cond,
        then_body,
        else_body,
        ..
    } = expr
    {
        let cond_val = eval_expr(cond, env, trace)?;
        let truthy = match &cond_val {
            Value::Array(a) if a.rank() == 0 => a.data()[0] != 0.0,
            Value::Result { ok, .. } => *ok,
            other => {
                return Err(EvalError::Unsupported(format!(
                    "if condition must be a scalar or Result, got {}",
                    value_kind(other)
                )));
            }
        };
        let body = if truthy { then_body } else { else_body };
        let mut last = Value::Array(DenseArray::from_scalar(0.0));
        for stmt in body {
            last = eval_expr(stmt, env, trace)?;
        }
        return Ok(last);
    }
    if let Expr::While { cond, body, .. } = expr {
        return crate::eval_loop::eval_while(cond, body, env, trace);
    }
    if let Expr::TryCatch {
        body,
        binding,
        handler,
        ..
    } = expr
    {
        return crate::result_ops::eval_try_catch(body, binding, handler, env, trace);
    }
    if let Expr::Break { value, .. } = expr {
        let v = match value {
            Some(inner) => eval_expr(inner, env, trace)?,
            None => Value::Array(DenseArray::from_scalar(0.0)),
        };
        return Err(EvalError::BreakSignal(Box::new(v)));
    }
    if matches!(expr, Expr::Continue { .. }) {
        return Err(EvalError::ContinueSignal);
    }
    if let Expr::FnDef {
        name,
        params,
        body,
        annotations,
        span,
    } = expr
    {
        let source = env
            .pending_source
            .as_ref()
            .and_then(|s| s.get(span.start..span.end))
            .map(str::to_string);
        env.define_fn(
            name.clone(),
            mlpl_eval_state::UserFn::new(params.clone(), body.clone())
                .with_source(source)
                .with_annotations(annotations.clone()),
        );
        crate::def_metadata::register_def(name, annotations, span, env, trace)?;
        return Ok(Value::Array(DenseArray::from_scalar(0.0)));
    }
    if let Expr::Return { value, .. } = expr {
        let v = match value {
            Some(inner) => eval_expr(inner, env, trace)?,
            None => Value::Array(DenseArray::from_scalar(0.0)),
        };
        return Err(EvalError::ReturnSignal(Box::new(v)));
    }
    let (op_name, inputs, result) = match expr {
        Expr::IntLit(n, _) => ("literal", vec![], DenseArray::from_scalar(*n as f64)),
        Expr::FloatLit(f, _) => ("literal", vec![], DenseArray::from_scalar(*f)),
        Expr::StrLit(_, _) => unreachable!(),
        Expr::BuiltinRef(_, _) => unreachable!(),
        Expr::Include(_, _) => unreachable!(),
        Expr::Ident(name, _) => {
            let r = env
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))?;
            ("ident", vec![], r)
        }
        // Saga 29 step 002: every ArrayLit is dispatched by the
        // early-return at the head of `eval_expr`, so this arm
        // never fires.
        Expr::ArrayLit(_, _) => unreachable!(),
        Expr::UnaryNeg { operand, .. } => {
            let val = eval_expr(operand, env, trace)?.into_array()?;
            let r = DenseArray::from_scalar(-1.0).apply_binop(&val, |a, b| a * b)?;
            ("negate", vec![TraceValue::from_array(&val)], r)
        }
        Expr::Assign { name, value, .. } => {
            let is_param_ctor = matches!(
                value.as_ref(),
                Expr::TensorCtor {
                    kind: TensorCtorKind::Param,
                    ..
                }
            );
            let v = eval_expr(value, env, trace)?;
            // Cross-kind shadowing: a fresh binding must clear the
            // name from every value table (stale-kind bug, 2026-08-05).
            env.clear_binding(name);
            match v {
                Value::Model(m) => {
                    env.models.insert(name.clone(), m);
                    let placeholder = DenseArray::from_scalar(0.0);
                    ("assign_model", vec![], placeholder)
                }
                Value::Tokenizer(t) => {
                    env.set_tokenizer(name.clone(), t);
                    let placeholder = DenseArray::from_scalar(0.0);
                    ("assign_tokenizer", vec![], placeholder)
                }
                Value::GenState(g) => {
                    env.gen_states.insert(name.clone(), *g);
                    let placeholder = DenseArray::from_scalar(0.0);
                    ("assign_gen_state", vec![], placeholder)
                }
                Value::Str(s) => {
                    env.set_string(name.clone(), s);
                    ("assign_string", vec![], DenseArray::from_scalar(0.0))
                }
                Value::BuiltinRef { name: target } | Value::UserFnRef { name: target } => {
                    env.set_builtin_ref(name.clone(), target);
                    ("assign_builtin_ref", vec![], DenseArray::from_scalar(0.0))
                }
                Value::Array(val) => {
                    env.set(name.clone(), val.clone());
                    if is_param_ctor {
                        env.mark_param(name);
                    }
                    if let Some(tag) = crate::auto_tag::for_assign(value, env) {
                        env.set_tag(name.clone(), tag);
                    } else if let Some(tag) = crate::tag_propagate::propagate(value, env)? {
                        env.set_tag(name.clone(), tag);
                    }
                    ("assign", vec![TraceValue::from_array(&val)], val)
                }
                Value::DeviceTensor { .. } => {
                    env.set_device_tensor(name.clone(), v.clone());
                    return Ok(v);
                }
                Value::Record { fields } => {
                    env.set_record(name.clone(), fields.clone());
                    return Ok(Value::Record { fields });
                }
                Value::StrList { items } => {
                    env.set_string_list(name.clone(), items.clone());
                    return Ok(Value::StrList { items });
                }
                Value::Result { ok, payload } => {
                    env.set_result(name.clone(), ok, (*payload).clone());
                    return Ok(Value::Result { ok, payload });
                }
            }
        }
        Expr::BinOp { op, lhs, rhs, .. } => eval_binop(op, lhs, rhs, env, trace)?,
        Expr::FnCall { name, args, .. } => eval_fncall(name, args, env, trace)?,
        Expr::TensorCtor { kind, shape, .. } => {
            crate::eval_blocks::eval_tensor_ctor(*kind, shape, env, trace)?
        }
        Expr::Repeat { count, body, .. } => {
            crate::eval_blocks::eval_repeat(count, body, env, trace)?
        }
        Expr::Train { count, body, .. } => crate::eval_blocks::eval_train(count, body, env, trace)?,
        Expr::For {
            binding,
            source,
            body,
            ..
        } => crate::eval_for::eval_for(binding, source, body, env, trace)?,
        Expr::Experiment { name, body, .. } => {
            crate::experiment::eval_experiment(name, body, env, trace)?
        }
        Expr::Device { .. } => unreachable!(),
        // Saga 29 step 001: RecordLit and FieldAccess are dispatched
        // by the early-return `if let` block at the head of
        // `eval_expr`. Saga 31 step 004: If is also early-return.
        Expr::RecordLit { .. }
        | Expr::FieldAccess { .. }
        | Expr::If { .. }
        | Expr::While { .. }
        | Expr::Break { .. }
        | Expr::Continue { .. }
        | Expr::FnDef { .. }
        | Expr::TryCatch { .. }
        | Expr::Return { .. } => unreachable!(),
    };
    if let Some(t) = trace.as_mut() {
        let seq = t.events().len() as u64;
        let (input_types, output_type) = crate::auto_tag::for_trace_event(expr, env);
        t.push(TraceEvent {
            seq,
            op: op_name.into(),
            span: expr.span(),
            inputs,
            output: TraceValue::from_array(&result),
            input_types,
            output_type,
        });
    }
    Ok(Value::Array(result))
}
