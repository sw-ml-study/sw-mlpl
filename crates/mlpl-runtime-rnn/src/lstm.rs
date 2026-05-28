use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_element::prelude::*;
use mlpl_array_ops_matmul::prelude::*;
use mlpl_runtime_core::error::RuntimeError;

pub(crate) fn lstm_cell(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 5 {
        return Err(RuntimeError::ArityMismatch {
            func: "lstm_cell".into(),
            expected: 5,
            got: args.len(),
        });
    }
    let hd = args[1].data().len();
    let combined = concat_inputs(&args[0], &args[1])?;
    let gates = args[3].matmul(&combined)?;
    let gates = gates.apply_binop(&args[4], |a, b| a + b)?;
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
    let out = apply_gates(g, args[2].data(), hd);
    Ok(DenseArray::new(Shape::new(vec![2 * hd, 1]), out)?)
}

fn concat_inputs(input: &DenseArray, hidden: &DenseArray) -> Result<DenseArray, RuntimeError> {
    let id = input.data().len();
    let hd = hidden.data().len();
    let mut combined = Vec::with_capacity(id + hd);
    combined.extend_from_slice(input.data());
    combined.extend_from_slice(hidden.data());
    Ok(DenseArray::new(Shape::new(vec![id + hd, 1]), combined)?)
}

fn apply_gates(g: &[f64], cell_data: &[f64], hd: usize) -> Vec<f64> {
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut new_hidden = vec![0.0; hd];
    let mut new_cell = vec![0.0; hd];
    for i in 0..hd {
        let ig = sigmoid(g[i]);
        let fg = sigmoid(g[hd + i]);
        let cg = g[2 * hd + i].tanh();
        let og = sigmoid(g[3 * hd + i]);
        new_cell[i] = fg * cell_data[i] + ig * cg;
        new_hidden[i] = og * new_cell[i].tanh();
    }
    new_hidden.extend_from_slice(&new_cell);
    new_hidden
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(dims: &[usize], data: Vec<f64>) -> DenseArray {
        DenseArray::new(Shape::new(dims.to_vec()), data).unwrap()
    }

    #[test]
    fn lstm_cell_basic() {
        let (hd, id) = (2, 2);
        let args = vec![
            col(&[id, 1], vec![1.0, 0.5]),
            col(&[hd, 1], vec![0.0; hd]),
            col(&[hd, 1], vec![0.0; hd]),
            col(&[4 * hd, id + hd], vec![0.1; 4 * hd * (id + hd)]),
            col(&[4 * hd, 1], vec![0.0; 4 * hd]),
        ];
        let d = lstm_cell(args).unwrap();
        assert_eq!(d.shape().dims(), &[2 * hd, 1]);
        let s = 1.0 / (1.0 + (-0.15_f64).exp());
        let new_c = s * 0.15_f64.tanh();
        let new_h = s * new_c.tanh();
        assert!((d.data()[0] - new_h).abs() < 1e-10);
        assert!((d.data()[2] - new_c).abs() < 1e-10);
    }
}
