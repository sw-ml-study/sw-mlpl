use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub(crate) const NAMES: &[&str] = &["rnn_cell"];

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "rnn_cell" => Some(rnn_cell(args)),
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
    let input = &args[0];
    let hidden = &args[1];
    let w_ih = &args[2];
    let w_hh = &args[3];
    let bias = &args[4];
    let ih = w_ih.matmul(input)?;
    let hh = w_hh.matmul(hidden)?;
    let sum = ih.apply_binop(&hh, |a, b| a + b)?;
    let biased = sum.apply_binop(bias, |a, b| a + b)?;
    Ok(biased.map(f64::tanh))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlpl_array::Shape;

    #[test]
    fn rnn_cell_basic() {
        let input = DenseArray::new(Shape::new(vec![2, 1]), vec![1.0, 0.5]).unwrap();
        let hidden = DenseArray::new(Shape::new(vec![3, 1]), vec![0.0; 3]).unwrap();
        let w_ih =
            DenseArray::new(Shape::new(vec![3, 2]), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let w_hh = DenseArray::new(Shape::new(vec![3, 3]), vec![0.0; 9]).unwrap();
        let bias = DenseArray::new(Shape::new(vec![3, 1]), vec![0.0; 3]).unwrap();
        let result = rnn_cell(vec![input, hidden, w_ih, w_hh, bias]).unwrap();
        assert_eq!(result.shape().dims(), &[3, 1]);
        let d = result.data();
        let expected = [
            (0.1 * 1.0 + 0.2 * 0.5_f64).tanh(),
            (0.3 * 1.0 + 0.4 * 0.5_f64).tanh(),
            (0.5 * 1.0 + 0.6 * 0.5_f64).tanh(),
        ];
        for i in 0..3 {
            assert!(
                (d[i] - expected[i]).abs() < 1e-10,
                "mismatch at {i}: {} vs {}",
                d[i],
                expected[i]
            );
        }
    }
}
