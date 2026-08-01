//! `engram_stats(e, ids)` / `engram_stats(e, h, ids)` -- the Engram
//! health panel (saga E3): addressing counters from the frozen hash
//! contract (`mlpl_engram_core::addressing_stats`), memory-table
//! health, and (three-argument form) gate activation via the eager
//! forward. Returns a record so fields are addressable:
//! `s.unique_rows`, `s.collisions`, `s.gate_mean`, ...

use std::collections::BTreeMap;

use mlpl_array::DenseArray;
use mlpl_engram_core::{AddressingStats, HashSpec, addressing_stats};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::model_apply_engram::{engram_forward, resolve_engram_model};
use crate::model_engram_math::param;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Shorthand: a scalar record field.
fn scalar(v: f64) -> Value {
    Value::Array(DenseArray::from_scalar(v))
}

/// Evaluate `engram_stats(e, ids)` or `engram_stats(e, h, ids)`.
///
/// # Errors
/// Wrong arity, a non-engram model, non-integral ids, or (3-arg
/// form) anything the eager forward rejects.
pub fn eval_engram_stats(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 && args.len() != 3 {
        return Err(EvalError::Unsupported(
            "engram_stats: takes (e, ids) or (e, h, ids)".into(),
        ));
    }
    let model = resolve_engram_model(&args[0], env)?;
    let eval = mlpl_eval_env::dispatch_hook::eval_or_err;
    let ids = eval(args.last().expect("arity checked"), env, trace)?.into_array()?;
    let (spec, id_vals) = stats_inputs(&model, &ids)?;
    let s = addressing_stats(&id_vals, &spec).map_err(|e| EvalError::Unsupported(e.to_string()))?;
    let mut fields = addressing_map(&s);
    fields.extend(memory_fields(&model, env)?);
    if args.len() == 3 {
        let h = eval(&args[1], env, trace)?.into_array()?;
        fields.extend(gate_fields(&model, &h, &ids, env)?);
    }
    Ok(Value::Record { fields })
}

/// Validate the model + ids into the frozen hash-contract inputs.
fn stats_inputs(model: &ModelSpec, ids: &DenseArray) -> Result<(HashSpec, Vec<u64>), EvalError> {
    let ModelSpec::Engram {
        ngram_orders,
        heads,
        slots,
        seed,
        ..
    } = model
    else {
        return Err(EvalError::Unsupported(
            "engram_stats: model is not an engram layer".into(),
        ));
    };
    let spec = HashSpec {
        ngram_orders: ngram_orders.clone(),
        heads_per_ngram: *heads,
        slots_per_head: *slots,
        seed: *seed,
    };
    Ok((spec, ids_as_u64(ids)?))
}

/// Every id as an exact non-negative integer, or a loud error.
fn ids_as_u64(ids: &DenseArray) -> Result<Vec<u64>, EvalError> {
    ids.data()
        .iter()
        .map(|&v| {
            (v >= 0.0 && v.fract() == 0.0)
                .then_some(v as u64)
                .ok_or_else(|| {
                    EvalError::Unsupported(format!(
                        "engram_stats: ids must be non-negative integers, got {v}"
                    ))
                })
        })
        .collect()
}

/// The addressing counters as record fields.
fn addressing_map(s: &AddressingStats) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("rows_addressed".into(), scalar(s.lookups as f64)),
        ("unique_rows".into(), scalar(s.unique_rows as f64)),
        ("collisions".into(), scalar(s.collisions as f64)),
    ])
}

/// Memory-table health: rows carrying any signal + max row L2 norm.
fn memory_fields(
    model: &ModelSpec,
    env: &Environment,
) -> Result<BTreeMap<String, Value>, EvalError> {
    let ModelSpec::Engram {
        memory, head_dim, ..
    } = model
    else {
        unreachable!("stats_inputs already checked the variant");
    };
    let table = param(env, memory)?;
    let (mut nonzero, mut max_norm_sq) = (0u64, 0.0f64);
    for row in table.data().chunks(*head_dim) {
        let norm_sq: f64 = row.iter().map(|v| v * v).sum();
        nonzero += u64::from(norm_sq > 0.0);
        max_norm_sq = max_norm_sq.max(norm_sq);
    }
    Ok(BTreeMap::from([
        ("nonzero_rows".into(), scalar(nonzero as f64)),
        ("max_row_norm".into(), scalar(max_norm_sq.sqrt())),
    ]))
}

/// Mean/max gate activation from the eager forward's gate output.
fn gate_fields(
    model: &ModelSpec,
    h: &DenseArray,
    ids: &DenseArray,
    env: &Environment,
) -> Result<BTreeMap<String, Value>, EvalError> {
    let (_, gate) = engram_forward(model, h, ids, env)?;
    let data = gate.data();
    let mean = data.iter().sum::<f64>() / data.len().max(1) as f64;
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(BTreeMap::from([
        ("gate_mean".into(), scalar(mean)),
        ("gate_max".into(), scalar(max)),
    ]))
}
