//! Op enums and forward/backward helpers for autograd nodes.

use mlpl_array::DenseArray;
use mlpl_array_ops_element::prelude::*;
use mlpl_tensor_handle::{BinKind, UnaryKind};

/// Unary elementwise op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// -x
    Neg,
    /// exp(x)
    Exp,
    /// natural log
    Log,
    /// max(0, x)
    Relu,
    /// tanh(x)
    Tanh,
    /// 1 / (1 + exp(-x))
    Sigmoid,
}

/// Binary elementwise op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// a + b
    Add,
    /// a - b
    Sub,
    /// a * b
    Mul,
    /// a / b
    Div,
}

impl UnaryOp {
    /// Forward pass: apply the op elementwise.
    #[must_use]
    pub fn forward(self, x: &DenseArray) -> DenseArray {
        let f: fn(f64) -> f64 = match self {
            Self::Neg => |v| -v,
            Self::Exp => f64::exp,
            Self::Log => f64::ln,
            Self::Relu => |v| v.max(0.0),
            Self::Tanh => f64::tanh,
            Self::Sigmoid => |v| 1.0 / (1.0 + (-v).exp()),
        };
        x.map(f)
    }

    /// Compute local gradient wrt input x given upstream gradient.
    ///
    /// `x` is the parent input; `y` is the forward output (cached).
    #[must_use]
    pub fn backward(self, x: &DenseArray, y: &DenseArray, upstream: &DenseArray) -> DenseArray {
        let n = upstream.data().len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x.data()[i];
            let yi = y.data()[i];
            let g = upstream.data()[i];
            let local = match self {
                Self::Neg => -1.0,
                Self::Exp => yi,
                Self::Log => 1.0 / xi,
                Self::Relu => {
                    if xi > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                Self::Tanh => 1.0 - yi * yi,
                Self::Sigmoid => yi * (1.0 - yi),
            };
            out.push(g * local);
        }
        DenseArray::new(upstream.shape().clone(), out).expect("shape preserved")
    }
}

impl BinaryOp {
    /// Forward pass with scalar broadcasting.
    pub fn forward(
        self,
        a: &DenseArray,
        b: &DenseArray,
    ) -> Result<DenseArray, mlpl_array::ArrayError> {
        let op: fn(f64, f64) -> f64 = match self {
            Self::Add => |x, y| x + y,
            Self::Sub => |x, y| x - y,
            Self::Mul => |x, y| x * y,
            Self::Div => |x, y| x / y,
        };
        a.apply_binop(b, op)
    }

    /// Compute upstream gradients for the two parents.
    ///
    /// Returns `(grad_a, grad_b)`, each shaped like the upstream grad
    /// (caller is responsible for un-broadcasting to the parent shape).
    #[must_use]
    pub fn backward(
        self,
        a_val: &DenseArray,
        b_val: &DenseArray,
        upstream: &DenseArray,
    ) -> (DenseArray, DenseArray) {
        // Broadcast each operand up to the output (upstream) shape so
        // per-element indexing is valid under FULL rank/axis broadcasting
        // (not just scalars); the caller then un-broadcasts each grad back
        // to its parent shape.
        let up_to_out = |v: &DenseArray| {
            if v.shape() == upstream.shape() {
                v.clone()
            } else {
                v.apply_binop(upstream, |x, _| x)
                    .expect("broadcast operand to output shape")
            }
        };
        let (a_b, b_b) = (up_to_out(a_val), up_to_out(b_val));
        let n = upstream.data().len();
        let mut ga = Vec::with_capacity(n);
        let mut gb = Vec::with_capacity(n);
        for i in 0..n {
            let (g, ai, bi) = (upstream.data()[i], a_b.data()[i], b_b.data()[i]);
            let (da, db) = match self {
                Self::Add => (g, g),
                Self::Sub => (g, -g),
                Self::Mul => (g * bi, g * ai),
                Self::Div => (g / bi, -g * ai / (bi * bi)),
            };
            ga.push(da);
            gb.push(db);
        }
        let shape = upstream.shape().clone();
        (
            DenseArray::new(shape.clone(), ga).expect("shape"),
            DenseArray::new(shape, gb).expect("shape"),
        )
    }
}

/// The tape's `UnaryOp` in device terms (moved from resident.rs
/// to honor the module function budget).
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
