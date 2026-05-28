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
