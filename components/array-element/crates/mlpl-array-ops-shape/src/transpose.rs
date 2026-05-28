use mlpl_array::{ArrayError, DenseArray, Shape, compute_strides};

/// Transpose extension for `DenseArray`.
pub trait TransposeExt {
    /// Transpose: reverse axis order, physically reordering the
    /// row-major buffer. Rank <= 1 is a no-op clone.
    #[must_use]
    fn transpose(&self) -> DenseArray;
}

impl TransposeExt for DenseArray {
    fn transpose(&self) -> DenseArray {
        let dims = self.shape().dims();
        if dims.len() <= 1 {
            return self.clone();
        }
        let new_dims: Vec<usize> = dims.iter().rev().copied().collect();
        let new_shape = Shape::new(new_dims);
        let new_data = permute_data(self.data(), dims, new_shape.dims());
        let arr = DenseArray::new(new_shape, new_data).expect("element count preserved");
        attach_reversed_labels(arr, self).expect("label count preserved")
    }
}

fn permute_data(src: &[f64], old_dims: &[usize], new_dims: &[usize]) -> Vec<f64> {
    let n = src.len();
    let rank = old_dims.len();
    let old_strides = compute_strides(old_dims);
    let new_strides = compute_strides(new_dims);
    let mut out = vec![0.0; n];
    for flat in 0..n {
        let mut remainder = flat;
        let mut new_flat = 0;
        for axis in 0..rank {
            let idx = remainder / old_strides[axis];
            remainder %= old_strides[axis];
            new_flat += idx * new_strides[rank - 1 - axis];
        }
        out[new_flat] = src[flat];
    }
    out
}

fn attach_reversed_labels(arr: DenseArray, src: &DenseArray) -> Result<DenseArray, ArrayError> {
    match src.labels() {
        Some(lbls) => arr.with_labels(lbls.iter().rev().cloned().collect()),
        None => Ok(arr),
    }
}
