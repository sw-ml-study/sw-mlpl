//! Input validation for `analysis::analysis_boundary_2d`.
//! Lifted into its own module so the renderer stays under the
//! 25-LOC function-LOC budget without pushing `analysis.rs` past
//! the 7-function ceiling (saga 33 step 025).

use mlpl_array::DenseArray;

use crate::svg::VizError;

pub(crate) fn validate(
    grid_outputs: &DenseArray,
    dims: &DenseArray,
    points: &DenseArray,
    labels: &DenseArray,
) -> Result<(usize, usize, usize), VizError> {
    let bad = |m: &str| Err(VizError::InvalidShape(m.into()));
    if dims.rank() != 1 || dims.data().len() != 2 {
        return bad("boundary_2d dims must be [rows, cols]");
    }
    let rows = dims.data()[0] as usize;
    let cols = dims.data()[1] as usize;
    if grid_outputs.data().len() != rows * cols {
        return bad("boundary_2d grid_outputs length does not match dims");
    }
    let pdims = points.shape().dims();
    if pdims.len() != 2 || pdims[1] != 2 {
        return bad("boundary_2d points must be Nx2");
    }
    let n = pdims[0];
    if labels.rank() != 1 || labels.data().len() != n {
        return bad("boundary_2d labels length must match points");
    }
    Ok((rows, cols, n))
}
