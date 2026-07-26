use mlpl_array::{ArrayError, DenseArray, Shape};

/// N-way concatenation along an existing axis. All inputs must
/// have identical shape. Output shape matches the input shape
/// with `dims[axis]` multiplied by `n`.
pub fn stack(arrays: &[&DenseArray], axis: usize) -> Result<DenseArray, ArrayError> {
    let first = validate(arrays, axis)?;
    let dims = first.shape().dims();
    let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let parent_stride = dims[axis] * inner;
    let outer: usize = dims[..axis].iter().product();
    let n = arrays.len();
    let mut out_dims = dims.to_vec();
    out_dims[axis] = n * dims[axis];
    let mut data: Vec<f64> = Vec::with_capacity(outer * n * parent_stride);
    for o in 0..outer {
        for arr in arrays {
            let start = o * parent_stride;
            data.extend_from_slice(&arr.data()[start..start + parent_stride]);
        }
    }
    let arr = DenseArray::new(Shape::new(out_dims), data)?;
    match first.labels() {
        Some(l) => arr.with_labels(l.to_vec()),
        None => Ok(arr),
    }
}

fn validate<'a>(arrays: &'a [&DenseArray], axis: usize) -> Result<&'a DenseArray, ArrayError> {
    let first = *arrays.first().ok_or(ArrayError::ShapeMismatch {
        source: 0,
        target: 1,
    })?;
    let dims = first.shape().dims();
    if axis >= dims.len() {
        return Err(ArrayError::ShapeMismatch {
            source: axis,
            target: dims.len(),
        });
    }
    for a in arrays.iter().skip(1) {
        if a.shape().dims() != dims {
            return Err(ArrayError::ShapeMismatch {
                source: a.shape().dims().len(),
                target: dims.len(),
            });
        }
    }
    Ok(first)
}

/// Cyclic-rotate extension for `DenseArray`. Lives beside `stack`
/// (whole-array recomposition family) to stay inside the module
/// budgets; re-exported from the crate prelude like its siblings.
pub trait RotateExt {
    /// APL-style cyclic rotate along `axis`: positive `k` brings
    /// element `k` to the front (left/up shift), negative `k`
    /// rotates the other way, any magnitude wraps. Shape and axis
    /// labels are preserved.
    fn rotate(&self, k: i64, axis: usize) -> Result<DenseArray, ArrayError>;
}

impl RotateExt for DenseArray {
    fn rotate(&self, k: i64, axis: usize) -> Result<DenseArray, ArrayError> {
        crate::take::validate(self, axis, 0)?;
        let dims = self.shape().dims();
        let n = dims[axis];
        let shift = k.rem_euclid(n as i64) as usize;
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
        let out = rotated_data(self.data(), outer, n, shift, inner);
        let arr = DenseArray::new(self.shape().clone(), out)?;
        match self.labels() {
            Some(l) => arr.with_labels(l.to_vec()),
            None => Ok(arr),
        }
    }
}

/// Copy `data` with the `axis` dimension cyclically shifted left by
/// `shift` (already reduced mod `n`).
fn rotated_data(data: &[f64], outer: usize, n: usize, shift: usize, inner: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len());
    for o in 0..outer {
        for i in 0..n {
            let src = (o * n + (i + shift) % n) * inner;
            out.extend_from_slice(&data[src..src + inner]);
        }
    }
    out
}
