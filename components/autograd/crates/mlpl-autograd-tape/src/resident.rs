//! Resident-forward attempts (saga E4 step 003). [`try_resident`]
//! returns `Some(handle)` when the op could run on the registered
//! device backend -- the tape node then carries the LAZY resident
//! result -- and `None` when the caller should take the CPU path
//! (residency off, no backend, both operands host-side after a
//! structural-op fallback, or a backend error; correctness first,
//! the CPU path is always right).

use mlpl_tensor_handle::{AxisKind, BinKind, TensorHandle, UnaryKind};

use crate::ops::{BinaryOp, UnaryOp};
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
}

/// The tape's `UnaryOp` in device terms.
#[must_use]
pub fn map_unary(op: UnaryOp) -> UnaryKind {
    match op {
        UnaryOp::Neg => UnaryKind::Neg,
        UnaryOp::Exp => UnaryKind::Exp,
        UnaryOp::Log => UnaryKind::Log,
        UnaryOp::Relu => UnaryKind::Relu,
        UnaryOp::Tanh => UnaryKind::Tanh,
        UnaryOp::Sigmoid => UnaryKind::Sigmoid,
    }
}

/// The tape's `BinaryOp` in device terms.
#[must_use]
pub fn map_binary(op: BinaryOp) -> BinKind {
    match op {
        BinaryOp::Add => BinKind::Add,
        BinaryOp::Sub => BinKind::Sub,
        BinaryOp::Mul => BinKind::Mul,
        BinaryOp::Div => BinKind::Div,
    }
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
        ResidentReq::Reshape(x, dims) => handle_of(tape, x)?.dev_reshape(dims).ok(),
        ResidentReq::Ce(x, targets) => handle_of(tape, x)?.dev_cross_entropy(targets).ok(),
    }
}
