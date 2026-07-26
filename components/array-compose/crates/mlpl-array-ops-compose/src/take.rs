use mlpl_array::{ArrayError, DenseArray, Shape};

/// Take extension for `DenseArray`.
pub trait TakeExt {
    /// Drop one axis at a single integer index.
    fn take(&self, axis: usize, idx: usize) -> Result<DenseArray, ArrayError>;
}

impl TakeExt for DenseArray {
    fn take(&self, axis: usize, idx: usize) -> Result<DenseArray, ArrayError> {
        validate(self, axis, idx)?;
        let dims = self.shape().dims();
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
        let out = collect(self.data(), outer, dims[axis], idx, inner);
        let new_dims = drop_at(dims.iter().copied(), axis);
        let arr = DenseArray::new(Shape::new(new_dims), out)?;
        match self.labels() {
            Some(lbls) => arr.with_labels(drop_at(lbls.iter().cloned(), axis)),
            None => Ok(arr),
        }
    }
}

pub(crate) fn validate(a: &DenseArray, axis: usize, idx: usize) -> Result<(), ArrayError> {
    let dims = a.shape().dims();
    if axis >= dims.len() {
        return Err(ArrayError::ShapeMismatch {
            source: axis,
            target: dims.len(),
        });
    }
    if idx >= dims[axis] {
        return Err(ArrayError::ShapeMismatch {
            source: idx,
            target: dims[axis],
        });
    }
    Ok(())
}

fn drop_at<T>(it: impl IntoIterator<Item = T>, axis: usize) -> Vec<T> {
    it.into_iter()
        .enumerate()
        .filter_map(|(k, v)| (k != axis).then_some(v))
        .collect()
}

fn collect(data: &[f64], outer: usize, axis_size: usize, idx: usize, inner: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        let start = (o * axis_size + idx) * inner;
        out.extend_from_slice(&data[start..start + inner]);
    }
    out
}
