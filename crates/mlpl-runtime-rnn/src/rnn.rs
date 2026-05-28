use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "rnn_cell" => Some(rnn_cell(args)),
        "lstm_cell" => Some(crate::lstm::lstm_cell(args)),
        _ => None,
    }
}

fn rnn_cell(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 5 {
        return Err(RuntimeError::ArityMismatch {
            func: "rnn_cell".into(),
            expected: 5,
            got: args.len(),
        });
    }
    let ih = args[2].matmul(&args[0])?;
    let hh = args[3].matmul(&args[1])?;
    let sum = ih.apply_binop(&hh, |a, b| a + b)?;
    let biased = sum.apply_binop(&args[4], |a, b| a + b)?;
    Ok(biased.map(f64::tanh))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlpl_array::Shape;

    fn col(dims: &[usize], data: Vec<f64>) -> DenseArray {
        DenseArray::new(Shape::new(dims.to_vec()), data).unwrap()
    }

    #[test]
    fn rnn_cell_basic() {
        let args = vec![
            col(&[2, 1], vec![1.0, 0.5]),
            col(&[3, 1], vec![0.0; 3]),
            col(&[3, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            col(&[3, 3], vec![0.0; 9]),
            col(&[3, 1], vec![0.0; 3]),
        ];
        let d = rnn_cell(args).unwrap();
        assert_eq!(d.shape().dims(), &[3, 1]);
        let expected = [
            (0.1 + 0.2 * 0.5_f64).tanh(),
            (0.3 + 0.4 * 0.5_f64).tanh(),
            (0.5 + 0.6 * 0.5_f64).tanh(),
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert!((d.data()[i] - e).abs() < 1e-10);
        }
    }
}
