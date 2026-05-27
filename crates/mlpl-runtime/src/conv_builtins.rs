use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

pub(crate) const NAMES: &[&str] = &["conv2d", "pool2d"];

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "conv2d" => Some(conv2d(args)),
        "pool2d" => Some(pool2d(args)),
        _ => None,
    }
}

fn conv2d(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 4 {
        return Err(RuntimeError::ArityMismatch {
            func: "conv2d".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let input = &args[0];
    let filters = &args[1];
    let stride = args[2].data()[0] as usize;
    let padding = args[3].data()[0] as usize;

    let id = input.shape().dims();
    let fd = filters.shape().dims();
    if id.len() != 4 || fd.len() != 4 {
        return Err(RuntimeError::InvalidArgument {
            func: "conv2d".into(),
            reason: format!(
                "input must be [B,C_in,H,W] and filters [C_out,C_in,kH,kW], got ranks {} and {}",
                id.len(),
                fd.len()
            ),
        });
    }
    let (b, c_in, h, w) = (id[0], id[1], id[2], id[3]);
    let (c_out, fc_in, kh, kw) = (fd[0], fd[1], fd[2], fd[3]);
    if c_in != fc_in {
        return Err(RuntimeError::InvalidArgument {
            func: "conv2d".into(),
            reason: format!("input channels {c_in} != filter channels {fc_in}"),
        });
    }
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let mut out = vec![0.0f64; b * c_out * h_out * w_out];
    let idata = input.data();
    let fdata = filters.data();

    for bi in 0..b {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0;
                    for ci in 0..c_in {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = oh * stride + fh;
                                let iw = ow * stride + fw;
                                let (ih, iw) = (
                                    ih as isize - padding as isize,
                                    iw as isize - padding as isize,
                                );
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let iv = idata[bi * c_in * h * w
                                        + ci * h * w
                                        + ih as usize * w
                                        + iw as usize];
                                    let fv =
                                        fdata[co * fc_in * kh * kw + ci * kh * kw + fh * kw + fw];
                                    sum += iv * fv;
                                }
                            }
                        }
                    }
                    out[bi * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow] = sum;
                }
            }
        }
    }
    Ok(DenseArray::new(
        Shape::new(vec![b, c_out, h_out, w_out]),
        out,
    )?)
}

fn pool2d(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: "pool2d".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let input = &args[0];
    let size = &args[1];
    let mode_arr = &args[2];

    let id = input.shape().dims();
    if id.len() != 4 {
        return Err(RuntimeError::InvalidArgument {
            func: "pool2d".into(),
            reason: format!("input must be [B,C,H,W], got rank {}", id.len()),
        });
    }
    let sd = size.data();
    if sd.len() != 2 {
        return Err(RuntimeError::InvalidArgument {
            func: "pool2d".into(),
            reason: format!("size must be [pH, pW], got {} elements", sd.len()),
        });
    }
    let (ph, pw) = (sd[0] as usize, sd[1] as usize);
    let (b, c, h, w) = (id[0], id[1], id[2], id[3]);
    let h_out = h / ph;
    let w_out = w / pw;

    let mode_val = mode_arr.data();
    let is_max = mode_val.first().copied().unwrap_or(0.0) != 0.0;

    let idata = input.data();
    let mut out = vec![0.0f64; b * c * h_out * w_out];
    for bi in 0..b {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut val = if is_max { f64::NEG_INFINITY } else { 0.0 };
                    for fh in 0..ph {
                        for fw in 0..pw {
                            let v = idata
                                [bi * c * h * w + ci * h * w + (oh * ph + fh) * w + ow * pw + fw];
                            if is_max {
                                val = val.max(v);
                            } else {
                                val += v;
                            }
                        }
                    }
                    if !is_max {
                        val /= (ph * pw) as f64;
                    }
                    out[bi * c * h_out * w_out + ci * h_out * w_out + oh * w_out + ow] = val;
                }
            }
        }
    }
    Ok(DenseArray::new(Shape::new(vec![b, c, h_out, w_out]), out)?)
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
        let result = conv2d(vec![input, filter, stride, padding]).unwrap();
        assert_eq!(result.shape().dims(), &[1, 1, 2, 2]);
        let d = result.data();
        assert_eq!(d[0], 45.0);
        assert_eq!(d[1], 54.0);
        assert_eq!(d[2], 81.0);
        assert_eq!(d[3], 90.0);
    }

    #[test]
    fn pool2d_max() {
        let input = DenseArray::new(
            Shape::new(vec![1, 1, 4, 4]),
            (0..16).map(|i| i as f64).collect(),
        )
        .unwrap();
        let size = DenseArray::new(Shape::new(vec![2]), vec![2.0, 2.0]).unwrap();
        let mode = DenseArray::from_scalar(1.0);
        let result = pool2d(vec![input, size, mode]).unwrap();
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
        let result = pool2d(vec![input, size, mode]).unwrap();
        assert_eq!(result.shape().dims(), &[1, 1, 2, 2]);
        assert_eq!(result.data(), &[2.5, 4.5, 10.5, 12.5]);
    }
}
