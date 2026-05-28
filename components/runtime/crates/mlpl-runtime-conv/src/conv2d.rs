use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

struct Params {
    b: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
}

pub(crate) fn run(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 4 {
        return Err(RuntimeError::ArityMismatch {
            func: "conv2d".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let p = validate(&args)?;
    let out = compute(args[0].data(), args[1].data(), &p);
    Ok(DenseArray::new(
        Shape::new(vec![p.b, p.c_out, p.h_out, p.w_out]),
        out,
    )?)
}

fn validate(args: &[DenseArray]) -> Result<Params, RuntimeError> {
    let (id, fd) = (args[0].shape().dims(), args[1].shape().dims());
    let err = |reason: String| RuntimeError::InvalidArgument {
        func: "conv2d".into(),
        reason,
    };
    if id.len() != 4 || fd.len() != 4 {
        return Err(err(format!(
            "input [B,C_in,H,W] and filters [C_out,C_in,kH,kW] \
             required, got ranks {} and {}",
            id.len(),
            fd.len()
        )));
    }
    if id[1] != fd[1] {
        return Err(err(format!(
            "input channels {} != filter channels {}",
            id[1], fd[1]
        )));
    }
    let (stride, padding) = (args[2].data()[0] as usize, args[3].data()[0] as usize);
    Ok(Params {
        b: id[0],
        c_in: id[1],
        h: id[2],
        w: id[3],
        c_out: fd[0],
        kh: fd[2],
        kw: fd[3],
        stride,
        padding,
        h_out: (id[2] + 2 * padding - fd[2]) / stride + 1,
        w_out: (id[3] + 2 * padding - fd[3]) / stride + 1,
    })
}

fn compute(idata: &[f64], fdata: &[f64], p: &Params) -> Vec<f64> {
    let size = p.b * p.c_out * p.h_out * p.w_out;
    let mut out = vec![0.0f64; size];
    for bi in 0..p.b {
        for co in 0..p.c_out {
            for oh in 0..p.h_out {
                for ow in 0..p.w_out {
                    out[bi * p.c_out * p.h_out * p.w_out
                        + co * p.h_out * p.w_out
                        + oh * p.w_out
                        + ow] = pixel(idata, fdata, p, bi, co, oh, ow);
                }
            }
        }
    }
    out
}

fn pixel(id: &[f64], fd: &[f64], p: &Params, bi: usize, co: usize, oh: usize, ow: usize) -> f64 {
    let mut sum = 0.0;
    for ci in 0..p.c_in {
        for fh in 0..p.kh {
            for fw in 0..p.kw {
                let ih = (oh * p.stride + fh) as isize - p.padding as isize;
                let iw = (ow * p.stride + fw) as isize - p.padding as isize;
                if ih < 0 || ih >= p.h as isize || iw < 0 || iw >= p.w as isize {
                    continue;
                }
                let iv =
                    id[bi * p.c_in * p.h * p.w + ci * p.h * p.w + ih as usize * p.w + iw as usize];
                let fv = fd[co * p.c_in * p.kh * p.kw + ci * p.kh * p.kw + fh * p.kw + fw];
                sum += iv * fv;
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv2d_basic() {
        let input = DenseArray::new(
            Shape::new(vec![1, 1, 4, 4]),
            (0..16).map(|i| i as f64).collect(),
        )
        .unwrap();
        let filter = DenseArray::new(Shape::new(vec![1, 1, 3, 3]), vec![1.0; 9]).unwrap();
        let stride = DenseArray::from_scalar(1.0);
        let padding = DenseArray::from_scalar(0.0);
        let result = run(vec![input, filter, stride, padding]).unwrap();
        assert_eq!(result.shape().dims(), &[1, 1, 2, 2]);
        let d = result.data();
        assert_eq!(d[0], 45.0);
        assert_eq!(d[1], 54.0);
        assert_eq!(d[2], 81.0);
        assert_eq!(d[3], 90.0);
    }
}
