//! Pure cross-entropy kernels (no tape): fused log-softmax + NLL forward
//! and its backward. Moved here from `mlpl-autograd`'s tensor_ops during the
//! autograd-partition so the backward pass owns them and the Tensor API crate
//! stays acyclic.

use mlpl_array::DenseArray;

/// Fused log-softmax + NLL forward pass. Scalar output.
///
/// `logits` is `[N, V]` (caller must pre-flatten `[B, T, V]` to
/// `[B*T, V]`); `targets` has length `N`. Stable via max-subtraction
/// inside the log-sum-exp.
pub fn cross_entropy_forward(logits: &DenseArray, targets: &[usize]) -> Result<DenseArray, String> {
    let (n, v) = ce_split_rows_cols(logits)?;
    if targets.len() != n {
        return Err(format!(
            "cross_entropy: target length {} does not match logits rows {}",
            targets.len(),
            n
        ));
    }
    let data = logits.data();
    let mut total = 0.0;
    for (i, &t) in targets.iter().enumerate() {
        if t >= v {
            return Err(format!(
                "cross_entropy: target index {t} out of range for V={v}"
            ));
        }
        let row = &data[i * v..(i + 1) * v];
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse: f64 = m + row.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        total += lse - row[t];
    }
    Ok(DenseArray::from_scalar(total / n as f64))
}

/// Cross-entropy backward pass: returns `d loss / d logits` shaped like
/// `logits`, computed as `(softmax(logits) - one_hot(targets)) / N *
/// upstream_scalar`.
pub fn cross_entropy_backward(
    logits: &DenseArray,
    targets: &[usize],
    upstream_scalar: f64,
) -> DenseArray {
    let (n, v) = ce_split_rows_cols(logits).expect("forward validated shape");
    let data = logits.data();
    let mut out = vec![0.0; data.len()];
    let inv_n = 1.0 / n as f64;
    for (i, &t) in targets.iter().enumerate() {
        let row = &data[i * v..(i + 1) * v];
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            let p = e / s;
            let indicator = if j == t { 1.0 } else { 0.0 };
            out[i * v + j] = upstream_scalar * (p - indicator) * inv_n;
        }
    }
    DenseArray::new(logits.shape().clone(), out).expect("shape preserved")
}

fn ce_split_rows_cols(logits: &DenseArray) -> Result<(usize, usize), String> {
    let dims = logits.shape().dims();
    match dims.len() {
        2 => Ok((dims[0], dims[1])),
        3 => Ok((dims[0] * dims[1], dims[2])),
        r => Err(format!(
            "cross_entropy: logits must be rank 2 or 3, got rank {r}"
        )),
    }
}
