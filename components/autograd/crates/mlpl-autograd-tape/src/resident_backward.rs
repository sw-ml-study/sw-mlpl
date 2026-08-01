//! Device-resident backward formulas (saga E4 step 004). Each
//! helper composes the gradient from lazy device ops and returns
//! `None` when the caller should run the exact CPU kernel instead
//! (no backend, unsupported rank/broadcast case, or any backend
//! error -- the CPU path is always correct). Formulas are chosen
//! to avoid constants where algebra allows: tanh' and sigmoid'
//! recompose without a ones tensor.

use mlpl_tensor_handle::{AxisKind, BinKind, TensorHandle, UnaryKind};

use crate::ops::{BinaryOp, UnaryOp};
use crate::resident::{as_dev, fill};

/// d(op(x))/dx pulled back through `up`. `y` is the forward output.
#[must_use]
pub fn unary_backward(
    op: UnaryOp,
    x: &TensorHandle,
    y: &TensorHandle,
    up: &TensorHandle,
) -> Option<TensorHandle> {
    let up = as_dev(up)?;
    match op {
        UnaryOp::Neg => up.dev_unary(UnaryKind::Neg).ok(),
        UnaryOp::Exp => up.dev_binary(BinKind::Mul, y).ok(),
        UnaryOp::Log => up.dev_binary(BinKind::Div, x).ok(),
        UnaryOp::Relu => {
            let mask = as_dev(x)?.dev_unary(UnaryKind::Gtz).ok()?;
            up.dev_binary(BinKind::Mul, &mask).ok()
        }
        UnaryOp::Tanh | UnaryOp::Sigmoid => smooth_backward(op, y, &up),
    }
}

/// tanh' = 1 - y^2 => up - up*y*y; sigmoid' = y(1-y) =>
/// up*y - (up*y)*y. Both constant-free recompositions.
fn smooth_backward(op: UnaryOp, y: &TensorHandle, up: &TensorHandle) -> Option<TensorHandle> {
    let base = match op {
        UnaryOp::Tanh => up.clone(),
        UnaryOp::Sigmoid => up.dev_binary(BinKind::Mul, y).ok()?,
        _ => return None,
    };
    let byy = base.dev_binary(BinKind::Mul, y).ok()?;
    let sub = match op {
        UnaryOp::Tanh => byy.dev_binary(BinKind::Mul, y).ok()?,
        _ => byy,
    };
    base.dev_binary(BinKind::Sub, &sub).ok()
}

/// Elementwise binary backward for the EQUAL-SHAPE case (the tape's
/// broadcasting cases keep the exact CPU unbroadcast path).
#[must_use]
pub fn binary_backward(
    op: BinaryOp,
    a: &TensorHandle,
    b: &TensorHandle,
    up: &TensorHandle,
) -> Option<(TensorHandle, TensorHandle)> {
    if a.dims() != up.dims() || b.dims() != up.dims() {
        return None;
    }
    let up = as_dev(up)?;
    match op {
        BinaryOp::Add => Some((up.clone(), up)),
        BinaryOp::Sub => Some((up.clone(), up.dev_unary(UnaryKind::Neg).ok()?)),
        BinaryOp::Mul => Some((
            up.dev_binary(BinKind::Mul, b).ok()?,
            up.dev_binary(BinKind::Mul, a).ok()?,
        )),
        BinaryOp::Div => div_backward(a, b, &up),
    }
}

/// d(a/b): `ga = up / b`, `gb = -(up * a) / b^2`.
fn div_backward(
    a: &TensorHandle,
    b: &TensorHandle,
    up: &TensorHandle,
) -> Option<(TensorHandle, TensorHandle)> {
    let ga = up.dev_binary(BinKind::Div, b).ok()?;
    let bb = as_dev(b)?.dev_binary(BinKind::Mul, b).ok()?;
    let gb = up
        .dev_binary(BinKind::Mul, a)
        .ok()?
        .dev_binary(BinKind::Div, &bb)
        .ok()?
        .dev_unary(UnaryKind::Neg)
        .ok()?;
    Some((ga, gb))
}

/// Matmul backward for the rank-2 x rank-2 case: `ga = up @ b^T`,
/// `gb = a^T @ up`. The matrix-vector case keeps the CPU kernel.
#[must_use]
pub fn matmul_backward(
    a: &TensorHandle,
    b: &TensorHandle,
    up: &TensorHandle,
) -> Option<(TensorHandle, TensorHandle)> {
    if b.dims().len() != 2 {
        return None;
    }
    let up = as_dev(up)?;
    let bt = as_dev(b)?.dev_unary(UnaryKind::Transpose).ok()?;
    let at = as_dev(a)?.dev_unary(UnaryKind::Transpose).ok()?;
    let ga = up.dev_binary(BinKind::Matmul, &bt).ok()?;
    let gb = at.dev_binary(BinKind::Matmul, &up).ok()?;
    Some((ga, gb))
}

/// SumAll / MeanAll backward: broadcast the upstream scalar over
/// the parent's shape (scaled by `1/n` for the mean).
#[must_use]
pub fn sum_mean_backward(
    parent_dims: &[usize],
    up: &TensorHandle,
    mean: bool,
) -> Option<TensorHandle> {
    let up = as_dev(up)?;
    let n: usize = parent_dims.iter().product();
    let scale = if mean { 1.0 / n as f64 } else { 1.0 };
    let base = fill(parent_dims, scale)?;
    base.dev_binary(BinKind::Mul, &up).ok()
}

/// Softmax backward: `g = y * (up - sum(up * y, axis, keep))`.
#[must_use]
pub fn softmax_backward(y: &TensorHandle, up: &TensorHandle, axis: usize) -> Option<TensorHandle> {
    let up = as_dev(up)?;
    let y = as_dev(y)?;
    let uy = up.dev_binary(BinKind::Mul, &y).ok()?;
    let t = uy.dev_axis(AxisKind::Sum, Some(axis), true).ok()?;
    let inner = up.dev_binary(BinKind::Sub, &t).ok()?;
    y.dev_binary(BinKind::Mul, &inner).ok()
}
