use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

struct Params {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    ph: usize,
    pw: usize,
    h_out: usize,
    w_out: usize,
    is_max: bool,
}

pub(crate) fn run(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: "pool2d".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let p = validate(&args)?;
    let out = compute(args[0].data(), &p);
    Ok(DenseArray::new(
        Shape::new(vec![p.b, p.c, p.h_out, p.w_out]),
        out,
    )?)
}

fn validate(args: &[DenseArray]) -> Result<Params, RuntimeError> {
    let err = |reason: String| RuntimeError::InvalidArgument {
        func: "pool2d".into(),
        reason,
    };
    let id = args[0].shape().dims();
    if id.len() != 4 {
        return Err(err(format!(
            "input must be [B,C,H,W], got rank {}",
            id.len()
        )));
    }
    let sd = args[1].data();
    if sd.len() != 2 {
        return Err(err(format!(
            "size must be [pH, pW], got {} elements",
            sd.len()
        )));
    }
    let (ph, pw) = (sd[0] as usize, sd[1] as usize);
    let is_max = args[2].data().first().copied().unwrap_or(0.0) != 0.0;
    Ok(Params {
        b: id[0],
        c: id[1],
        h: id[2],
        w: id[3],
        ph,
        pw,
        h_out: id[2] / ph,
        w_out: id[3] / pw,
        is_max,
    })
}

fn compute(idata: &[f64], p: &Params) -> Vec<f64> {
    let mut out = vec![0.0f64; p.b * p.c * p.h_out * p.w_out];
    for bi in 0..p.b {
        for ci in 0..p.c {
            for oh in 0..p.h_out {
                for ow in 0..p.w_out {
                    out[bi * p.c * p.h_out * p.w_out
                        + ci * p.h_out * p.w_out
                        + oh * p.w_out
                        + ow] = window(idata, p, bi, ci, oh, ow);
                }
            }
        }
    }
    out
}

fn window(idata: &[f64], p: &Params, bi: usize, ci: usize, oh: usize, ow: usize) -> f64 {
    let init = if p.is_max { f64::NEG_INFINITY } else { 0.0 };
    let mut val = init;
    for fh in 0..p.ph {
        for fw in 0..p.pw {
            let idx =
                bi * p.c * p.h * p.w + ci * p.h * p.w + (oh * p.ph + fh) * p.w + ow * p.pw + fw;
            if p.is_max {
                val = val.max(idata[idx]);
            } else {
                val += idata[idx];
            }
        }
    }
    if !p.is_max {
        val /= (p.ph * p.pw) as f64;
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool2d_max() {
        let input = DenseArray::new(
            Shape::new(vec![1, 1, 4, 4]),
            (0..16).map(|i| i as f64).collect(),
        )
        .unwrap();
        let size = DenseArray::new(Shape::new(vec![2]), vec![2.0, 2.0]).unwrap();
        let mode = DenseArray::from_scalar(1.0);
        let result = run(vec![input, size, mode]).unwrap();
        assert_eq!(result.shape().dims(), &[1, 1, 2, 2]);
        assert_eq!(result.data(), &[5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn pool2d_avg() {
        let input = DenseArray::new(
            Shape::new(vec![1, 1, 4, 4]),
            (0..16).map(|i| i as f64).collect(),
        )
        .unwrap();
        let size = DenseArray::new(Shape::new(vec![2]), vec![2.0, 2.0]).unwrap();
        let mode = DenseArray::from_scalar(0.0);
        let result = run(vec![input, size, mode]).unwrap();
        assert_eq!(result.shape().dims(), &[1, 1, 2, 2]);
        assert_eq!(result.data(), &[2.5, 4.5, 10.5, 12.5]);
    }
}
