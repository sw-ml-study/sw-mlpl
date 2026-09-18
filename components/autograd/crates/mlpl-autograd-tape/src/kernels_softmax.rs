//! Softmax forward/backward kernels + the gradient accumulator.
//! Split from `ops.rs` (tech-debt spike step 002).

use mlpl_array::DenseArray;
use mlpl_array_ops_element::prelude::*;

/// Split `dims` into `(outer, axis_len, inner)` for lane iteration along
/// `axis`: in row-major order element `(o, k, i)` lives at
/// `o*axis_len*inner + k*inner + i`. A softmax lane is the `axis_len`
/// elements sharing an `(o, i)` pair.
#[must_use]
fn axis_strides(dims: &[usize], axis: usize) -> (usize, usize, usize) {
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    (outer, dims[axis], inner)
}

/// Numerically-stable softmax along `axis`, for any rank (attention needs
/// a non-last axis and rank-3 batches).
#[must_use]
pub fn softmax_forward(x: &DenseArray, axis: usize) -> DenseArray {
    let dims = x.shape().dims();
    if dims.is_empty() {
        return x.clone();
    }
    let (outer, axis_len, inner) = axis_strides(dims, axis);
    let data = x.data();
    let mut out = vec![0.0; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| o * axis_len * inner + k * inner + i;
            let m = (0..axis_len)
                .map(|k| data[at(k)])
                .fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = (0..axis_len).map(|k| (data[at(k)] - m).exp()).collect();
            let s: f64 = exps.iter().sum();
            for (k, e) in exps.iter().enumerate() {
                out[at(k)] = e / s;
            }
        }
    }
    DenseArray::new(x.shape().clone(), out).expect("shape")
}

/// Softmax backward given forward output `y` and upstream grad `g`.
///
/// For each row: `dx_i = y_i * (g_i - sum_j(g_j * y_j))`.
#[must_use]
pub fn softmax_backward(y: &DenseArray, upstream: &DenseArray, axis: usize) -> DenseArray {
    let dims = y.shape().dims();
    if dims.is_empty() {
        return DenseArray::from_scalar(0.0);
    }
    let (outer, axis_len, inner) = axis_strides(dims, axis);
    let (yd, gd) = (y.data(), upstream.data());
    let mut out = vec![0.0; yd.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| o * axis_len * inner + k * inner + i;
            // g_k = y_k * (up_k - sum_j up_j y_j) along the axis lane.
            let dot: f64 = (0..axis_len).map(|k| yd[at(k)] * gd[at(k)]).sum();
            for k in 0..axis_len {
                out[at(k)] = yd[at(k)] * (gd[at(k)] - dot);
            }
        }
    }
    DenseArray::new(y.shape().clone(), out).expect("shape")
}

/// Accumulate `incoming` into `slot`: add if already present, else
/// set. Handles every residency mix (saga E4 step 004): two host
/// grads use the exact f64 add as always; when either side is
/// device-resident the add runs on the backend (the host side
/// uploads), falling back to the exact host add on backend errors.
pub fn accumulate(
    slot: &mut Option<mlpl_tensor_handle::TensorHandle>,
    incoming: impl Into<mlpl_tensor_handle::TensorHandle>,
) {
    use mlpl_tensor_handle::{BinKind, TensorHandle};
    let incoming = incoming.into();
    match slot {
        None => *slot = Some(incoming),
        Some(existing) => {
            if (existing.is_dev() || incoming.is_dev())
                && let Ok(sum) = existing.dev_binary(BinKind::Add, &incoming)
            {
                *existing = sum;
                return;
            }
            let sum = existing
                .to_dense()
                .apply_binop(&incoming.to_dense(), |a, b| a + b)
                .expect("matching shapes during accumulation");
            *existing = TensorHandle::Cpu(sum);
        }
    }
}

/// Accumulate a (left, right) gradient pair -- the shared tail of
/// every two-parent backward rule.
pub fn accumulate_pair(
    tape: &crate::tape::Tape,
    left: crate::tape::NodeId,
    right: crate::tape::NodeId,
    ga: impl Into<mlpl_tensor_handle::TensorHandle>,
    gb: impl Into<mlpl_tensor_handle::TensorHandle>,
) {
    let mut nodes = tape.nodes_mut();
    accumulate(&mut nodes[left.0].grad, ga);
    accumulate(&mut nodes[right.0].grad, gb);
}

/// Seed `node`'s gradient with ones (saga E4: resident roots seed a
/// device-side fill so backward stays lazy; host roots stay
/// bit-exact f64). No-op when a gradient is already present.
pub fn seed_ones(node: &mut crate::tape::NodeData) {
    if node.grad.is_some() {
        return;
    }
    let dims = node.value.dims();
    node.grad = Some(if node.value.is_dev() {
        crate::resident::fill(&dims, 1.0).unwrap_or_else(|| host_ones(&dims))
    } else {
        host_ones(&dims)
    });
}

/// Host-side f64 ones of `dims`.
fn host_ones(dims: &[usize]) -> mlpl_tensor_handle::TensorHandle {
    let n = dims.iter().product();
    mlpl_tensor_handle::TensorHandle::Cpu(
        DenseArray::new(mlpl_array::Shape::new(dims.to_vec()), vec![1.0; n])
            .expect("ones fill matches dims"),
    )
}
