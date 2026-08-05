//! Name -> op dispatch for the array-valued kg builtins.
//! `kg_split` (record-valued) is bound at the eval layer; the
//! entity-partition core it uses lives in `paths.rs`.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::RuntimeError;

use crate::edges;

/// Dispatch the array-valued kg builtins; `None` if not matched.
pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "kg_neighbors" => Some(kg_neighbors(name, &args)),
        "kg_verify" => Some(kg_verify(name, &args)),
        "kg_paths" => Some(crate::paths::kg_paths(name, &args)),
        _ => None,
    }
}

/// `kg_neighbors(edges, node[, rel])` -- sorted unique one-hop
/// destination ids from `node`, optionally along one relation.
fn kg_neighbors(name: &str, args: &[DenseArray]) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 && args.len() != 3 {
        return Err(arity(name, 2, args.len()));
    }
    let e = edges::parse(name, &args[0])?;
    let node = args[1].data()[0] as i64;
    let rel = args.get(2).map(|r| r.data()[0] as i64);
    let ids: Vec<f64> = e
        .neighbors(node, rel)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    Ok(DenseArray::from_vec(ids))
}

/// `kg_verify(edges, paths)` -- row-batched path checking: for an
/// `[n, L]` array of id sequences, `out[i] = 1.0` iff every
/// consecutive pair in row i is an edge (any relation). Rank-1
/// input is one path.
fn kg_verify(name: &str, args: &[DenseArray]) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity(name, 2, args.len()));
    }
    let e = edges::parse(name, &args[0])?;
    let p = &args[1];
    let (n, l) = match p.shape().dims() {
        [l] => (1, *l),
        [n, l] => (*n, *l),
        d => {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("paths must be rank 1 or [n, L], got shape {d:?}"),
            });
        }
    };
    let ok_row = |row: &[f64]| row.windows(2).all(|w| e.connects(w[0] as i64, w[1] as i64));
    let data: Vec<f64> = (0..n)
        .map(|i| f64::from(ok_row(&p.data()[i * l..(i + 1) * l])))
        .collect();
    Ok(DenseArray::from_vec(data))
}

pub(crate) fn arity(name: &str, expected: usize, got: usize) -> RuntimeError {
    RuntimeError::ArityMismatch {
        func: name.into(),
        expected,
        got,
    }
}

/// Rebuild an `[E, 3]` array from parsed triples (split output).
pub(crate) fn edges_array(rows: &[(i64, i64, i64)]) -> Result<DenseArray, RuntimeError> {
    let mut data = Vec::with_capacity(rows.len() * 3);
    for (s, r, d) in rows {
        data.extend_from_slice(&[*s as f64, *r as f64, *d as f64]);
    }
    Ok(DenseArray::new(Shape::new(vec![rows.len(), 3]), data)?)
}
