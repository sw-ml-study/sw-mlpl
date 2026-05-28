use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

/// `cross_entropy(logits, targets)` fused log-softmax + NLL,
/// scalar mean. Logits are `[N, V]` or `[B, T, V]` float;
/// targets are `[N]` or `[B, T]` integer-valued.
pub(crate) fn cross_entropy(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let (n, v) = validate_logits_shape(name, &args[0])?;
    let idx = collect_target_indices(name, &args[1], n, v)?;
    let data = args[0].data();
    let mut total = 0.0;
    for (i, &t) in idx.iter().enumerate() {
        let row = &data[i * v..(i + 1) * v];
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse: f64 = m + row.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        total += lse - row[t];
    }
    Ok(DenseArray::from_scalar(total / n as f64))
}

/// `perplexity(logits, targets) = exp(cross_entropy(logits, targets))`.
pub(crate) fn perplexity(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    cross_entropy(name, args).map(|ce| DenseArray::from_scalar(ce.data()[0].exp()))
}

fn validate_logits_shape(name: &str, logits: &DenseArray) -> Result<(usize, usize), RuntimeError> {
    let dims = logits.shape().dims();
    match dims.len() {
        2 => Ok((dims[0], dims[1])),
        3 => Ok((dims[0] * dims[1], dims[2])),
        r => Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("logits must be rank 2 or 3, got rank {r}"),
        }),
    }
}

fn collect_target_indices(
    name: &str,
    targets: &DenseArray,
    n: usize,
    v: usize,
) -> Result<Vec<usize>, RuntimeError> {
    if targets.elem_count() != n {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!(
                "targets must have {n} elements to match logits rows, got {}",
                targets.elem_count()
            ),
        });
    }
    let mut idx = Vec::with_capacity(n);
    for (i, &t) in targets.data().iter().enumerate() {
        if t < 0.0 || t.fract() != 0.0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("target[{i}] must be a non-negative integer, got {t}"),
            });
        }
        let ti = t as usize;
        if ti >= v {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("target[{i}] = {ti} out of range for V = {v}"),
            });
        }
        idx.push(ti);
    }
    Ok(idx)
}
