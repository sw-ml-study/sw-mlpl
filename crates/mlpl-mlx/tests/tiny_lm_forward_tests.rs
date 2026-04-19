//! Saga 14 step 003 closing test: a Tiny LM forward pass routed
//! through the MLX runtime produces the same loss as the same
//! forward routed through `mlpl-rt`, within fp32 tolerance.
//!
//! The forward exercises every primitive the Saga 13 Tiny LM
//! `demos/tiny_lm.mlpl` uses through to its loss scalar:
//!
//!     embed -> positional -> rms_norm -> causal_attention
//!           -> residual -> rms_norm -> linear -> relu -> linear
//!           -> residual -> rms_norm -> linear -> cross_entropy
//!
//! The model is a single head, single block, scaled down to keep
//! the test cheap (V=8, d=4, T=4, ff_hidden=8) but structurally
//! identical to the demo. A `Backend` function-pointer table lets
//! the same forward run on either runtime; passing the test means
//! the MLX path produces the same loss as the CPU path on the
//! same fixed-seed parameters.
//!
//! Same triple-gate as the rest of `mlpl-mlx`: macOS + aarch64 +
//! `mlx` feature.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_mlx::{ArrayError, DenseArray, Shape};

/// Tolerance for the end-to-end forward parity. The Tiny LM
/// forward chains ~10 fp32-round-tripped MLX kernels; the
/// per-element error compounds roughly linearly. 5e-4 is loose
/// enough to absorb that compounding while still catching real
/// regressions.
const TINY_LM_TOL: f64 = 5e-4;

/// Per-runtime function pointers for every primitive the forward
/// touches. Pure Rust scaffolding (slice_cols, rms_norm by hand,
/// mask construction) lives outside the table so it is shared
/// across runtimes verbatim.
struct Backend {
    matmul: fn(&DenseArray, &DenseArray) -> Result<DenseArray, ArrayError>,
    add: fn(&DenseArray, &DenseArray) -> Result<DenseArray, ArrayError>,
    softmax: fn(&DenseArray, usize) -> Result<DenseArray, ArrayError>,
    transpose: fn(&DenseArray) -> DenseArray,
    relu: fn(&DenseArray) -> DenseArray,
    cross_entropy: fn(&DenseArray, &DenseArray) -> Result<DenseArray, ArrayError>,
}

/// CPU matmul wrapper -- `mlpl-rt` does not surface a free
/// `matmul` symbol because the compile-to-rust codegen lowers it
/// to a direct `DenseArray::matmul` method call. The MLX side does
/// have a free `mlpl_mlx::matmul`, so we wrap the CPU side here to
/// give the `Backend` table a uniform signature.
fn cpu_matmul(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    a.matmul(b)
}

const MLPL_RT: Backend = Backend {
    matmul: cpu_matmul,
    add: mlpl_rt::add,
    softmax: mlpl_rt::softmax,
    transpose: mlpl_rt::transpose,
    relu: mlpl_rt::relu,
    cross_entropy: mlpl_rt::cross_entropy,
};

const MLPL_MLX: Backend = Backend {
    matmul: mlpl_mlx::matmul,
    add: mlpl_mlx::add,
    softmax: mlpl_mlx::softmax,
    transpose: mlpl_mlx::transpose,
    relu: mlpl_mlx::relu,
    cross_entropy: mlpl_mlx::cross_entropy,
};

#[test]
fn tiny_lm_forward_loss_matches_cpu_within_tolerance() {
    let p = Params::new();
    let cpu_loss = forward_loss(&MLPL_RT, &p).unwrap();
    let mlx_loss = forward_loss(&MLPL_MLX, &p).unwrap();
    assert_eq!(cpu_loss.shape().dims(), &[] as &[usize]);
    assert_eq!(mlx_loss.shape().dims(), &[] as &[usize]);
    let diff = (cpu_loss.data()[0] - mlx_loss.data()[0]).abs();
    assert!(
        diff <= TINY_LM_TOL,
        "tiny LM forward loss diverges: cpu={} mlx={} diff={} tol={TINY_LM_TOL}",
        cpu_loss.data()[0],
        mlx_loss.data()[0],
        diff
    );
}

/// Tiny LM hyperparameters and frozen parameters. All weights are
/// produced by a deterministic PRNG so both backends see exactly
/// the same inputs.
struct Params {
    embed: DenseArray,   // [V, d]
    pos: DenseArray,     // [T, d]
    wq: DenseArray,      // [d, d]
    wk: DenseArray,      // [d, d]
    wv: DenseArray,      // [d, d]
    wo: DenseArray,      // [d, d]
    w1: DenseArray,      // [d, ff]
    w2: DenseArray,      // [ff, d]
    w_out: DenseArray,   // [d, V]
    tokens: DenseArray,  // [T] -- input ids as f64
    targets: DenseArray, // [T] -- next-token ids as f64
}

const V: usize = 8;
const D: usize = 4;
const T: usize = 4;
const FF: usize = 8;

impl Params {
    fn new() -> Self {
        let mut rng = 0x9E37_79B9_u32;
        let mut next = || {
            // splitmix32-ish: deterministic, dependency-free.
            rng = rng.wrapping_mul(0x85EB_CA6B).wrapping_add(0xC2B2_AE35);
            ((rng >> 8) as f64) / (1u32 << 24) as f64 - 0.5
        };
        let mat = |rows: usize, cols: usize, sample: &mut dyn FnMut() -> f64| {
            let data: Vec<f64> = (0..rows * cols).map(|_| sample()).collect();
            DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
        };
        let embed = mat(V, D, &mut next);
        let pos = mat(T, D, &mut next);
        let wq = mat(D, D, &mut next);
        let wk = mat(D, D, &mut next);
        let wv = mat(D, D, &mut next);
        let wo = mat(D, D, &mut next);
        let w1 = mat(D, FF, &mut next);
        let w2 = mat(FF, D, &mut next);
        let w_out = mat(D, V, &mut next);
        let tokens = DenseArray::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let targets = DenseArray::from_vec(vec![2.0, 4.0, 6.0, 0.0]);
        Self {
            embed,
            pos,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w_out,
            tokens,
            targets,
        }
    }
}

/// Run the full Tiny LM forward and return the cross_entropy loss.
fn forward_loss(b: &Backend, p: &Params) -> Result<DenseArray, ArrayError> {
    let onehot = tokens_to_onehot(&p.tokens);
    let emb = (b.matmul)(&onehot, &p.embed)?;
    let h0 = (b.add)(&emb, &p.pos)?;

    let attn_in = rms_norm(&h0);
    let attn_out = causal_attention(b, &attn_in, &p.wq, &p.wk, &p.wv, &p.wo)?;
    let h1 = (b.add)(&h0, &attn_out)?;

    let ff_in = rms_norm(&h1);
    let ff_h = (b.matmul)(&ff_in, &p.w1)?;
    let ff_act = (b.relu)(&ff_h);
    let ff_out = (b.matmul)(&ff_act, &p.w2)?;
    let h2 = (b.add)(&h1, &ff_out)?;

    let final_norm = rms_norm(&h2);
    let logits = (b.matmul)(&final_norm, &p.w_out)?;
    (b.cross_entropy)(&logits, &p.targets)
}

/// `[T]` integer ids -> `[T, V]` one-hot matrix (row-major).
fn tokens_to_onehot(tokens: &DenseArray) -> DenseArray {
    let n = tokens.shape().dims()[0];
    let mut data = vec![0.0_f64; n * V];
    for (row, &t) in tokens.data().iter().enumerate() {
        let id = t as usize;
        data[row * V + id] = 1.0;
    }
    DenseArray::new(Shape::new(vec![n, V]), data).unwrap()
}

/// Per-row RMS norm: `y[i, :] = x[i, :] / sqrt(mean(x[i, :]^2) + eps)`.
/// Pure Rust so it is identical for both backends; the parity test
/// is about MLX vs CPU on shared scaffolding, not about RMS norm
/// going through MLX.
fn rms_norm(x: &DenseArray) -> DenseArray {
    let dims = x.shape().dims().to_vec();
    let rows = dims[0];
    let cols = dims[1];
    let eps = 1e-8;
    let src = x.data();
    let mut out = Vec::with_capacity(src.len());
    for r in 0..rows {
        let row = &src[r * cols..(r + 1) * cols];
        let mean_sq: f64 = row.iter().map(|v| v * v).sum::<f64>() / cols as f64;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for v in row {
            out.push(v * scale);
        }
    }
    DenseArray::new(Shape::new(dims), out).unwrap()
}

/// Single-head causal attention: Q = x @ Wq, K = x @ Wk, V = x @ Wv,
/// scores = (Q @ K^T) * (1/sqrt(d)) + causal_mask, attn = softmax,
/// out = attn @ V @ Wo.
fn causal_attention(
    b: &Backend,
    x: &DenseArray,
    wq: &DenseArray,
    wk: &DenseArray,
    wv: &DenseArray,
    wo: &DenseArray,
) -> Result<DenseArray, ArrayError> {
    let q = (b.matmul)(x, wq)?;
    let k = (b.matmul)(x, wk)?;
    let v = (b.matmul)(x, wv)?;
    let kt = (b.transpose)(&k);
    let scores = (b.matmul)(&q, &kt)?;
    let scale = 1.0 / (D as f64).sqrt();
    let mut masked = Vec::with_capacity(T * T);
    for r in 0..T {
        for c in 0..T {
            let s = scores.data()[r * T + c];
            masked.push(if c > r { -1.0e9 } else { s * scale });
        }
    }
    let masked_arr = DenseArray::new(Shape::new(vec![T, T]), masked)?;
    let attn = (b.softmax)(&masked_arr, 1)?;
    let head_out = (b.matmul)(&attn, &v)?;
    (b.matmul)(&head_out, wo)
}
