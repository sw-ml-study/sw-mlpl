use mlpl_array::{ArrayError, DenseArray, Shape};

/// Concat extension for `DenseArray`.
pub trait ConcatExt {
    /// Concat two arrays along `axis`. Both inputs must agree on
    /// every dim except `axis`, where the sizes add.
    fn concat(&self, other: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError>;
}

impl ConcatExt for DenseArray {
    fn concat(&self, other: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
        validate(self, other, axis)?;
        let a_dims = self.shape().dims();
        let b_dims = other.shape().dims();
        let mut out_dims = a_dims.to_vec();
        out_dims[axis] = a_dims[axis] + b_dims[axis];
        let mut data = Vec::with_capacity(out_dims.iter().product());
        copy_rows(&mut data, self, other, axis);
        let arr = DenseArray::new(Shape::new(out_dims), data)?;
        match self.labels() {
            Some(l) => arr.with_labels(l.to_vec()),
            None => Ok(arr),
        }
    }
}

fn validate(a: &DenseArray, b: &DenseArray, axis: usize) -> Result<(), ArrayError> {
    let (a_dims, b_dims) = (a.shape().dims(), b.shape().dims());
    if a_dims.len() != b_dims.len() {
        return Err(ArrayError::ShapeMismatch {
            source: a_dims.len(),
            target: b_dims.len(),
        });
    }
    if axis >= a_dims.len() {
        return Err(ArrayError::ShapeMismatch {
            source: axis,
            target: a_dims.len(),
        });
    }
    for (k, (&x, &y)) in a_dims.iter().zip(b_dims.iter()).enumerate() {
        if k != axis && x != y {
            return Err(ArrayError::ShapeMismatch {
                source: x,
                target: y,
            });
        }
    }
    Ok(())
}

fn copy_rows(out: &mut Vec<f64>, a: &DenseArray, b: &DenseArray, axis: usize) {
    let (a_dims, b_dims) = (a.shape().dims(), b.shape().dims());
    let outer: usize = a_dims[..axis].iter().product();
    let inner: usize = a_dims[axis + 1..].iter().product::<usize>().max(1);
    let (a_slab, b_slab) = (a_dims[axis] * inner, b_dims[axis] * inner);
    for o in 0..outer {
        out.extend_from_slice(&a.data()[o * a_slab..o * a_slab + a_slab]);
        out.extend_from_slice(&b.data()[o * b_slab..o * b_slab + b_slab]);
    }
}
