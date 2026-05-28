//! `array_lit` primitive: mirrors the interpreter's `eval_array_lit`
//! so that `[1, 2, 3]` and `[[1,2],[3,4]]` lower to the same runtime
//! values the REPL produces.
//!
//! Semantics match `mlpl_eval::eval_ops::eval_array_lit`:
//! - empty list -> empty rank-1 array
//! - all elements rank-0 -> pack scalars into a rank-1 array
//! - otherwise -> stack rows, prepending the row count as the new
//!   outermost dimension.

use mlpl_array::{DenseArray, Shape};

use crate::error::Result;

/// Build an array from a list of sub-arrays. Scalars promote to a
/// flat vector; rank >= 1 elements stack into a (rows+1)-D result.
/// Returns `ArrayError` if sub-arrays have inconsistent shapes.
pub fn array_lit(elems: Vec<DenseArray>) -> Result {
    if elems.is_empty() {
        return Ok(DenseArray::from_vec(vec![]));
    }
    if elems.iter().all(|a| a.rank() == 0) {
        let data: Vec<f64> = elems.iter().map(|a| a.data()[0]).collect();
        return Ok(DenseArray::from_vec(data));
    }
    let inner_shape = elems[0].shape().clone();
    let rows = elems.len();
    let mut data = Vec::with_capacity(rows * inner_shape.elem_count());
    for a in &elems {
        data.extend_from_slice(a.data());
    }
    let mut dims = vec![rows];
    dims.extend_from_slice(inner_shape.dims());
    DenseArray::new(Shape::new(dims), data)
}
