//! Op enums and forward/backward helpers for autograd nodes.

use mlpl_array::DenseArray;

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

/// Numerically-stable softmax along `axis`, supports rank-1 and rank-2.
#[must_use]
pub fn softmax_forward(x: &DenseArray, axis: usize) -> DenseArray {
    let row = |r: &[f64]| -> Vec<f64> {
        let m = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = r.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / s).collect()
    };
    let dims = x.shape().dims().to_vec();
    match dims.len() {
        0 => x.clone(),
        1 => DenseArray::new(x.shape().clone(), row(x.data())).expect("shape"),
        2 => {
            let (rows, cols) = (dims[0], dims[1]);
            let mut out = vec![0.0; rows * cols];
            if axis == 1 {
                for r in 0..rows {
                    let sm = row(&x.data()[r * cols..(r + 1) * cols]);
                    out[r * cols..(r + 1) * cols].copy_from_slice(&sm);
                }
            } else {
                for c in 0..cols {
                    let col: Vec<f64> = (0..rows).map(|r| x.data()[r * cols + c]).collect();
                    let sm = row(&col);
                    for (r, v) in sm.iter().enumerate() {
                        out[r * cols + c] = *v;
                    }
                }
            }
            DenseArray::new(x.shape().clone(), out).expect("shape")
        }
        _ => panic!("softmax supports rank <= 2"),
    }
}

/// Softmax backward given forward output `y` and upstream grad `g`.
///
/// For each row: `dx_i = y_i * (g_i - sum_j(g_j * y_j))`.
#[must_use]
pub fn softmax_backward(y: &DenseArray, upstream: &DenseArray, axis: usize) -> DenseArray {
    let row_bw = |yr: &[f64], gr: &[f64]| -> Vec<f64> {
        let dot: f64 = yr.iter().zip(gr.iter()).map(|(a, b)| a * b).sum();
        yr.iter()
            .zip(gr.iter())
            .map(|(yi, gi)| yi * (gi - dot))
            .collect()
    };
    let dims = y.shape().dims().to_vec();
    match dims.len() {
        0 => DenseArray::from_scalar(0.0),
        1 => DenseArray::new(y.shape().clone(), row_bw(y.data(), upstream.data())).expect("shape"),
        2 => {
            let (rows, cols) = (dims[0], dims[1]);
            let mut out = vec![0.0; rows * cols];
            if axis == 1 {
                for r in 0..rows {
                    let dx = row_bw(
                        &y.data()[r * cols..(r + 1) * cols],
                        &upstream.data()[r * cols..(r + 1) * cols],
                    );
                    out[r * cols..(r + 1) * cols].copy_from_slice(&dx);
                }
            } else {
                for c in 0..cols {
                    let yc: Vec<f64> = (0..rows).map(|r| y.data()[r * cols + c]).collect();
                    let gc: Vec<f64> = (0..rows).map(|r| upstream.data()[r * cols + c]).collect();
                    let dx = row_bw(&yc, &gc);
                    for (r, v) in dx.iter().enumerate() {
                        out[r * cols + c] = *v;
                    }
                }
            }
            DenseArray::new(y.shape().clone(), out).expect("shape")
        }
        _ => panic!("softmax supports rank <= 2"),
    }
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
