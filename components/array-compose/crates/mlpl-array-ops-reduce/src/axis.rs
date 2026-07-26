use mlpl_array::{ArrayError, DenseArray, Shape, compute_strides};

/// Per-axis traversal parameters shared by reduce_axis and argmax_axis.
pub(crate) struct AxisWalk {
    pub(crate) result_shape: Shape,
    pub(crate) axis_size: usize,
    pub(crate) axis_stride: usize,
    pub(crate) result_labels: Option<Vec<Option<String>>>,
}

/// Validate `axis` and compute the result shape, traversal strides,
/// and label-removed label set. Returns an `AxisWalk` consumers
/// pass to their inner-loop body.
pub(crate) fn axis_walk(arr: &DenseArray, axis: usize) -> Result<AxisWalk, ArrayError> {
    let dims = arr.shape().dims();
    if axis >= dims.len() {
        return Err(ArrayError::IndexOutOfBounds {
            axis,
            index: axis,
            size: dims.len(),
        });
    }
    let mut result_dims: Vec<usize> = dims.to_vec();
    result_dims.remove(axis);
    let strides = compute_strides(dims);
    let result_labels = arr.labels().map(|lbls| {
        let mut out: Vec<Option<String>> = lbls.to_vec();
        out.remove(axis);
        out
    });
    Ok(AxisWalk {
        result_shape: Shape::new(result_dims),
        axis_size: dims[axis],
        axis_stride: strides[axis],
        result_labels,
    })
}

/// Flat-index -> result-flat-index, stripping the `axis` coordinate
/// from the row-major position.
pub(crate) fn project(flat: usize, w: &AxisWalk) -> usize {
    if w.axis_stride > 1 {
        let outer = flat / (w.axis_size * w.axis_stride);
        let inner = flat % w.axis_stride;
        outer * w.axis_stride + inner
    } else {
        flat / w.axis_size
    }
}

pub(crate) fn finish(data: Vec<f64>, w: AxisWalk) -> Result<DenseArray, ArrayError> {
    let arr = DenseArray::new(w.result_shape, data)?;
    match w.result_labels {
        Some(l) => arr.with_labels(l),
        None => Ok(arr),
    }
}
