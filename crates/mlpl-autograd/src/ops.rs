//! Op enums and forward/backward helpers for autograd nodes.

use mlpl_array::{DenseArray, Shape};

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
        let n = upstream.data().len();
        let mut ga = Vec::with_capacity(n);
        let mut gb = Vec::with_capacity(n);
        for i in 0..n {
            let g = upstream.data()[i];
            // Index parents by broadcast: scalars reuse index 0.
            let ai = if a_val.data().len() == 1 {
                a_val.data()[0]
            } else {
                a_val.data()[i]
            };
            let bi = if b_val.data().len() == 1 {
                b_val.data()[0]
            } else {
                b_val.data()[i]
            };
            match self {
                Self::Add => {
                    ga.push(g);
                    gb.push(g);
                }
                Self::Sub => {
                    ga.push(g);
                    gb.push(-g);
                }
                Self::Mul => {
                    ga.push(g * bi);
                    gb.push(g * ai);
                }
                Self::Div => {
                    ga.push(g / bi);
                    gb.push(-g * ai / (bi * bi));
                }
            }
        }
        let shape = upstream.shape().clone();
        (
            DenseArray::new(shape.clone(), ga).expect("shape"),
            DenseArray::new(shape, gb).expect("shape"),
        )
    }
}

/// Sum `grad` (shaped like the broadcasted result) down to `target_shape`
/// (the parent's original shape). Handles the scalar-broadcast case.
#[must_use]
pub fn unbroadcast(grad: DenseArray, target_shape: &Shape) -> DenseArray {
    if grad.shape() == target_shape {
        return grad;
    }
    // target was a scalar that got broadcast to grad.shape(); sum it.
    if target_shape.rank() == 0 {
        let s: f64 = grad.data().iter().sum();
        return DenseArray::from_scalar(s);
    }
    // Shapes should match otherwise; keep a safe fallback.
    grad
}

/// Accumulate `incoming` into `slot`: add if already present, else set.
pub fn accumulate(slot: &mut Option<DenseArray>, incoming: DenseArray) {
    match slot {
        None => *slot = Some(incoming),
        Some(existing) => {
            let sum = existing
                .apply_binop(&incoming, |a, b| a + b)
                .expect("matching shapes during accumulation");
            *existing = sum;
        }
    }
}
