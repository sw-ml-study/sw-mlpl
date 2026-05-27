use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub(crate) const NAMES: &[&str] = &["rnn_cell", "lstm_cell"];

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "rnn_cell" => Some(rnn_cell(args)),
        "lstm_cell" => Some(lstm_cell(args)),
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

/// lstm_cell(input, hidden, cell, W, bias)
/// W: [4*hd, id+hd], bias: [4*hd, 1]
/// Returns [2*hd, 1] = concat(new_hidden, new_cell)
fn lstm_cell(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 5 {
        return Err(RuntimeError::ArityMismatch {
            func: "lstm_cell".into(),
            expected: 5,
            got: args.len(),
        });
    }
    let input = &args[0];
    let hidden = &args[1];
    let cell = &args[2];
    let w = &args[3];
    let bias = &args[4];

    let id = input.data().len();
    let hd = hidden.data().len();
    let mut combined = Vec::with_capacity(id + hd);
    combined.extend_from_slice(input.data());
    combined.extend_from_slice(hidden.data());
    let combined = DenseArray::new(mlpl_array::Shape::new(vec![id + hd, 1]), combined)?;

    let gates = w.matmul(&combined)?;
    let gates = gates.apply_binop(bias, |a, b| a + b)?;
    let g = gates.data();

    if g.len() != 4 * hd {
        return Err(RuntimeError::InvalidArgument {
            func: "lstm_cell".into(),
            reason: format!(
                "W should produce 4*hidden_dim={} gates, got {}",
                4 * hd,
                g.len()
            ),
        });
    }

    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut new_cell = vec![0.0; hd];
    let mut new_hidden = vec![0.0; hd];
    let cd = cell.data();
    for i in 0..hd {
        let ig = sigmoid(g[i]);
        let fg = sigmoid(g[hd + i]);
        let cg = g[2 * hd + i].tanh();
        let og = sigmoid(g[3 * hd + i]);
        new_cell[i] = fg * cd[i] + ig * cg;
        new_hidden[i] = og * new_cell[i].tanh();
    }

    let mut out = new_hidden;
    out.extend_from_slice(&new_cell);
    Ok(DenseArray::new(
        mlpl_array::Shape::new(vec![2 * hd, 1]),
        out,
    )?)
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

    #[test]
    fn lstm_cell_basic() {
        let hd = 2;
        let id = 2;
        let input = DenseArray::new(Shape::new(vec![id, 1]), vec![1.0, 0.5]).unwrap();
        let hidden = DenseArray::new(Shape::new(vec![hd, 1]), vec![0.0; hd]).unwrap();
        let cell = DenseArray::new(Shape::new(vec![hd, 1]), vec![0.0; hd]).unwrap();
        let w = DenseArray::new(
            Shape::new(vec![4 * hd, id + hd]),
            vec![0.1; 4 * hd * (id + hd)],
        )
        .unwrap();
        let bias = DenseArray::new(Shape::new(vec![4 * hd, 1]), vec![0.0; 4 * hd]).unwrap();
        let result = lstm_cell(vec![input, hidden, cell, w, bias]).unwrap();
        assert_eq!(result.shape().dims(), &[2 * hd, 1]);
        let d = result.data();
        assert_eq!(d.len(), 4);
        // With all weights=0.1, input=[1,0.5], hidden=[0,0]:
        // combined=[1,0.5,0,0], W@combined = 0.1*(1+0.5) = 0.15 per gate
        // All 4 gates get 0.15: ig=sig(0.15), fg=sig(0.15), cg=tanh(0.15), og=sig(0.15)
        let sig015 = 1.0 / (1.0 + (-0.15_f64).exp());
        let tanh015 = 0.15_f64.tanh();
        let new_c = sig015 * 0.0 + sig015 * tanh015;
        let new_h = sig015 * new_c.tanh();
        assert!((d[0] - new_h).abs() < 1e-10);
        assert!((d[2] - new_c).abs() < 1e-10);
    }
}
