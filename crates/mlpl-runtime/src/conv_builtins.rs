use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

pub(crate) const NAMES: &[&str] = &["conv2d"];

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "conv2d" => Some(conv2d(args)),
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
}
