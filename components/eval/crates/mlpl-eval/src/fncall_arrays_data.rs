//! Array-valued builtins that load or reshape row data -- `load_images`,
//! `dedupe_rows`, `kg_split`. Split out of `fncall_arrays` so each module
//! stays small; `fncall_arrays::try_dispatch` routes here.

use mlpl_array::{ArrayError, DenseArray};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub(crate) fn eval_load_images(args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    let err = |detail: &str| EvalError::Unsupported(format!("load_images: {detail}"));
    let [a0, a1] = args else {
        return Err(EvalError::BadArity {
            func: "load_images".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let Expr::StrLit(dir, _) = a0 else {
        return Err(err("arg 0 must be a directory string"));
    };
    let Expr::ArrayLit(dims, _) = a1 else {
        return Err(err("arg 1 must be a [H, W] array literal"));
    };
    let [h, w] = dims.as_slice() else {
        return Err(err("expected exactly two [H, W] entries"));
    };
    let parse_dim = |e: &Expr| match e {
        Expr::IntLit(n, _) if *n >= 0 => Ok(*n as usize),
        _ => Err(err("[H, W] entries must be non-negative integers")),
    };
    crate::loader::eval_load_images(env, dir, parse_dim(h)?, parse_dim(w)?)
}

/// `dedupe_rows(X)` -- unique rows of a rank-2 array (first
/// occurrence kept, original order) as `{rows, index}`: `rows`
/// for direct use, `index` so companion arrays follow via
/// `gather_rows(Y, d.index)`.
pub(crate) fn eval_dedupe_rows(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    crate::grad::arity_check(args, 1, "dedupe_rows")?;
    let x = eval_expr(&args[0], env, trace)?.into_array()?;
    let dims = x.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(EvalError::Unsupported(format!(
            "dedupe_rows: input must be rank 2 [n, L], got shape {dims:?}"
        )));
    }
    let (rows, index) = dedupe_core(&x, dims[0], dims[1])?;
    let fields = std::collections::BTreeMap::from([
        ("rows".to_string(), Value::Array(rows)),
        ("index".to_string(), Value::Array(index)),
    ]);
    Ok(Value::Record { fields })
}

/// First-occurrence unique rows + their indices (bitwise row equality).
fn dedupe_core(x: &DenseArray, n: usize, l: usize) -> Result<(DenseArray, DenseArray), ArrayError> {
    let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
    let mut keep: Vec<usize> = Vec::new();
    for i in 0..n {
        let row = &x.data()[i * l..(i + 1) * l];
        if seen.insert(row.iter().map(|v| v.to_bits()).collect()) {
            keep.push(i);
        }
    }
    let mut rows = Vec::with_capacity(keep.len() * l);
    for &i in &keep {
        rows.extend_from_slice(&x.data()[i * l..(i + 1) * l]);
    }
    let rows = DenseArray::new(mlpl_array::Shape::new(vec![keep.len(), l]), rows)?;
    let index = DenseArray::from_vec(keep.into_iter().map(|i| i as f64).collect());
    Ok((rows, index))
}

/// `kg_split(edges, frac, seed)` -- entity-disjoint `{seen, unseen}`
/// split of an `[E, 3]` edge array: `unseen` edges touch entities
/// the `seen` side never contains (`train`/`eval` would collide
/// with the train keyword); core in mlpl-forge-kg.
pub(crate) fn eval_kg_split(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    crate::grad::arity_check(args, 3, "kg_split")?;
    let edges = eval_expr(&args[0], env, trace)?.into_array()?;
    let frac = eval_expr(&args[1], env, trace)?.into_array()?.data()[0];
    let seed = eval_expr(&args[2], env, trace)?.into_array()?.data()[0] as u64;
    let (train, evl) = mlpl_forge_kg::split_edges("kg_split", &edges, frac, seed)
        .map_err(|e| EvalError::Unsupported(e.to_string()))?;
    let fields = std::collections::BTreeMap::from([
        ("seen".to_string(), Value::Array(train)),
        ("unseen".to_string(), Value::Array(evl)),
    ]);
    Ok(Value::Record { fields })
}
