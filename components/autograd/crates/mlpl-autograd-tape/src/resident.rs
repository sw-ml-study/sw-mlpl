//! Resident-forward attempts (saga E4 step 003). [`try_resident`]
//! returns `Some(handle)` when the op could run on the registered
//! device backend -- the tape node then carries the LAZY resident
//! result -- and `None` when the caller should take the CPU path
//! (residency off, no backend, both operands host-side after a
//! structural-op fallback, or a backend error; correctness first,
//! the CPU path is always right).

use mlpl_tensor_handle::{AxisKind, BinKind, TensorHandle, UnaryKind};

use crate::tape::{NodeId, Tape};

/// One resident-forward request, in device terms.
pub enum ResidentReq<'a> {
    /// Elementwise unary or transpose on one parent.
    Unary(NodeId, UnaryKind),
    /// Elementwise binary or matmul on two parents.
    Binary(NodeId, NodeId, BinKind),
    /// Softmax / reduction (axis `None` = over all elements).
    Axis(NodeId, AxisKind, Option<usize>, bool),
    /// Reshape one parent to new dims.
    Reshape(NodeId, &'a [usize]),
    /// Fused cross-entropy forward over class indices.
    Ce(NodeId, &'a [usize]),
    /// Concat two parents along an axis.
    Concat(NodeId, NodeId, usize),
}

/// A node's handle, if this tape is in resident mode.
fn handle_of(tape: &Tape, id: NodeId) -> Option<TensorHandle> {
    tape.resident
        .get()
        .then(|| tape.nodes()[id.0].value.clone())
}

/// A device handle for `h`, uploading a host value when a backend
/// is registered. `None` when there is no backend (or upload fails).
#[must_use]
pub fn as_dev(h: &TensorHandle) -> Option<TensorHandle> {
    match h {
        TensorHandle::Dev(_) => Some(h.clone()),
        TensorHandle::Cpu(a) => mlpl_tensor_handle::upload(a).ok(),
    }
}

/// A resident array of `dims` filled with `value` (gradient seeds,
/// fill-style backward formulas). `None` without a backend.
#[must_use]
pub fn fill(dims: &[usize], value: f64) -> Option<TensorHandle> {
    let ops = mlpl_tensor_handle::device_ops()?;
    ops.full(dims, value).ok().map(TensorHandle::Dev)
}

/// Attempt `req` on the device backend. `None` = take the CPU path.
#[must_use]
pub fn try_resident(tape: &Tape, req: ResidentReq<'_>) -> Option<TensorHandle> {
    match req {
        ResidentReq::Unary(x, op) => handle_of(tape, x)?.dev_unary(op).ok(),
        ResidentReq::Binary(a, b, op) => {
            let hb = handle_of(tape, b)?;
            handle_of(tape, a)?.dev_binary(op, &hb).ok()
        }
        ResidentReq::Axis(x, op, axis, keep) => handle_of(tape, x)?.dev_axis(op, axis, keep).ok(),
        ResidentReq::Concat(a, b, axis) => {
            let hb = handle_of(tape, b)?;
            handle_of(tape, a)?.dev_concat(&hb, axis).ok()
        }
        ResidentReq::Reshape(x, dims) => handle_of(tape, x)?.dev_reshape(dims).ok(),
        ResidentReq::Ce(x, targets) => handle_of(tape, x)?.dev_cross_entropy(targets).ok(),
    }
}

/// Transpose backward: `g = up^T`, lazily on the device.
#[must_use]
pub fn transpose_backward(tape: &Tape, up: &TensorHandle) -> Option<TensorHandle> {
    if !tape.resident.get() {
        return None;
    }
    as_dev(up)?.dev_unary(UnaryKind::Transpose).ok()
}

/// Reshape backward: `g = reshape(up, orig)`, lazily on the device.
#[must_use]
pub fn reshape_backward(tape: &Tape, up: &TensorHandle, orig: &[usize]) -> Option<TensorHandle> {
    if !tape.resident.get() {
        return None;
    }
    as_dev(up)?.dev_reshape(orig).ok()
}

/// Reduce a broadcast gradient back to `target` dims on the device:
/// sum (keepdims) over every axis the forward broadcast expanded,
/// then reshape. `None` (unsupported layout or backend error) means
/// "run the exact CPU kernel".
#[must_use]
pub fn unbroadcast(g: &TensorHandle, target: &[usize]) -> Option<TensorHandle> {
    let gd = g.dims();
    if gd == target {
        return Some(g.clone());
    }
    let pad = gd.len().checked_sub(target.len())?;
    let mut out = g.clone();
    for (i, gdim) in gd.iter().enumerate() {
        let want = if i < pad { 1 } else { target[i - pad] };
        if want == 1 && *gdim != 1 {
            out = out.dev_axis(AxisKind::Sum, Some(i), true).ok()?;
        }
    }
    out.dev_reshape(target).ok()
}
