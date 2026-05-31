//! On-device Adam optimizer: moment buffers live as `mlx_rs::Array`
//! and every update is an MLX elementwise op, so optimizer state
//! never round-trips to the CPU.

use mlx_rs::Array;
use mlx_rs::error::Result;

/// Adam with first/second moment buffers held on-device (one pair per
/// parameter). Buffers init lazily on the first [`step`](Self::step)
/// so the caller need not know parameter shapes up front.
pub struct MlxAdam {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    t: i32,
    m: Vec<Array>,
    v: Vec<Array>,
}

impl MlxAdam {
    /// Adam with explicit hyperparameters.
    pub fn new(lr: f32, b1: f32, b2: f32, eps: f32) -> Self {
        Self {
            lr,
            b1,
            b2,
            eps,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Adam with the conventional defaults (b1=0.9, b2=0.999, eps=1e-8).
    pub fn with_lr(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8)
    }

    /// Apply one Adam update, in place, to every `params[i]` from its
    /// `grads[i]`. All arithmetic stays on-device.
    pub fn step(&mut self, params: &mut [Array], grads: &[Array]) -> Result<()> {
        if self.m.is_empty() {
            for g in grads {
                self.m.push(g.multiply(Array::from_f32(0.0))?);
                self.v.push(g.multiply(Array::from_f32(0.0))?);
            }
        }
        self.t += 1;
        for i in 0..params.len() {
            let (mut m, mut v) = (self.m[i].clone(), self.v[i].clone());
            params[i] = self.update_one(&params[i], &grads[i], &mut m, &mut v)?;
            self.m[i] = m;
            self.v[i] = v;
        }
        Ok(())
    }

    /// The per-parameter Adam math: update moments, bias-correct, step.
    fn update_one(&self, p: &Array, g: &Array, m: &mut Array, v: &mut Array) -> Result<Array> {
        *m = m
            .multiply(Array::from_f32(self.b1))?
            .add(g.multiply(Array::from_f32(1.0 - self.b1))?)?;
        *v = v
            .multiply(Array::from_f32(self.b2))?
            .add(g.square()?.multiply(Array::from_f32(1.0 - self.b2))?)?;
        let mhat = m.divide(Array::from_f32(1.0 - self.b1.powi(self.t)))?;
        let vhat = v.divide(Array::from_f32(1.0 - self.b2.powi(self.t)))?;
        let denom = vhat.sqrt()?.add(Array::from_f32(self.eps))?;
        let update = mhat.divide(denom)?.multiply(Array::from_f32(self.lr))?;
        p.subtract(update)
    }
}
