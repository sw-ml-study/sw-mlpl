//! Generalized dyadic transpose (APL2's `P transpose A`) on flat
//! arrays: `perm` is a 0-based axis permutation and result axis
//! `i` draws from source axis `perm[i]`. `transpose` is the
//! special case `perm = [rank-1, ..., 1, 0]`.

use mlpl_array::{ArrayError, DenseArray, Shape, compute_strides};

/// Axis-permutation extension for `DenseArray`.
pub trait TransposeAxesExt {
    /// Permute axes: result axis `i` is source axis `perm[i]`.
    /// `perm` must name every source axis exactly once.
    fn transpose_axes(&self, perm: &[usize]) -> Result<DenseArray, ArrayError>;
}

impl TransposeAxesExt for DenseArray {
    fn transpose_axes(&self, perm: &[usize]) -> Result<DenseArray, ArrayError> {
        let dims = self.shape().dims();
        let rank = dims.len();
        validate_perm(perm, rank)?;
        let new_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        let new_shape = Shape::new(new_dims);
        let data = permute(self.data(), dims, new_shape.dims(), perm);
        let arr = DenseArray::new(new_shape, data)?;
        match self.labels() {
            Some(lbls) => arr.with_labels(perm.iter().map(|&p| lbls[p].clone()).collect()),
            None => Ok(arr),
        }
    }
}

fn validate_perm(perm: &[usize], rank: usize) -> Result<(), ArrayError> {
    if perm.len() != rank {
        return Err(ArrayError::RankMismatch {
            expected: rank,
            got: perm.len(),
        });
    }
    let mut seen = vec![false; rank];
    for (i, &p) in perm.iter().enumerate() {
        if p >= rank || seen[p] {
            return Err(ArrayError::IndexOutOfBounds {
                axis: i,
                index: p,
                size: rank,
            });
        }
        seen[p] = true;
    }
    Ok(())
}

/// Reorder the row-major buffer: walk source flat indices,
/// decompose against the source strides, and scatter each value
/// to the result position whose axis `i` coordinate is the
/// source coordinate on axis `perm[i]`.
fn permute(src: &[f64], old_dims: &[usize], new_dims: &[usize], perm: &[usize]) -> Vec<f64> {
    let rank = old_dims.len();
    let old_strides = compute_strides(old_dims);
    let new_strides = compute_strides(new_dims);
    let mut out = vec![0.0; src.len()];
    for (flat, &v) in src.iter().enumerate() {
        let mut remainder = flat;
        let mut coords = vec![0usize; rank];
        for axis in 0..rank {
            coords[axis] = remainder / old_strides[axis];
            remainder %= old_strides[axis];
        }
        let new_flat: usize = (0..rank).map(|i| coords[perm[i]] * new_strides[i]).sum();
        out[new_flat] = v;
    }
    out
}
