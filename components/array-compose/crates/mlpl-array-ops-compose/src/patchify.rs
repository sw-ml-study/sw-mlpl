use mlpl_array::{ArrayError, DenseArray, Shape};

/// Patchify extension for `DenseArray` (ViT patch embedding rearrangement).
pub trait PatchifyExt {
    /// Take a `[B, C, H, W]` image batch and a square patch size `P`
    /// (which must divide both `H` and `W`). Return `[B, N, P*P*C]`
    /// where `N = (H/P) * (W/P)` and each row of the trailing axis is
    /// one patch flattened with channel-outer order:
    /// `c * P*P + dy * P + dx`.
    fn patchify(&self, patch_size: usize) -> Result<DenseArray, ArrayError>;
}

struct P {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    p: usize,
    nh: usize,
    nw: usize,
    n_patches: usize,
    patch_len: usize,
}

impl PatchifyExt for DenseArray {
    fn patchify(&self, patch_size: usize) -> Result<DenseArray, ArrayError> {
        let p = validate(self, patch_size)?;
        let out = compute(self.data(), &p);
        DenseArray::new(Shape::new(vec![p.b, p.n_patches, p.patch_len]), out)
    }
}

fn validate(a: &DenseArray, ps: usize) -> Result<P, ArrayError> {
    let dims = a.shape().dims();
    let bad_shape = |s, t| ArrayError::ShapeMismatch {
        source: s,
        target: t,
    };
    if dims.len() != 4 {
        return Err(bad_shape(dims.len(), 4));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    if ps == 0 || h % ps != 0 || w % ps != 0 {
        return Err(bad_shape(ps, h.min(w)));
    }
    Ok(P {
        b,
        c,
        h,
        w,
        p: ps,
        nh: h / ps,
        nw: w / ps,
        n_patches: (h / ps) * (w / ps),
        patch_len: ps * ps * c,
    })
}

fn compute(src: &[f64], p: &P) -> Vec<f64> {
    let mut out = vec![0.0; p.b * p.n_patches * p.patch_len];
    for b_i in 0..p.b {
        for i in 0..p.nh {
            for j in 0..p.nw {
                copy_patch(src, &mut out, p, b_i, i, j, i * p.nw + j);
            }
        }
    }
    out
}

fn copy_patch(src: &[f64], out: &mut [f64], p: &P, b_i: usize, i: usize, j: usize, n: usize) {
    for c_i in 0..p.c {
        for dy in 0..p.p {
            for dx in 0..p.p {
                let src_idx = ((b_i * p.c + c_i) * p.h + i * p.p + dy) * p.w + j * p.p + dx;
                let dst_idx =
                    (b_i * p.n_patches + n) * p.patch_len + c_i * p.p * p.p + dy * p.p + dx;
                out[dst_idx] = src[src_idx];
            }
        }
    }
}
