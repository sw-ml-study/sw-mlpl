//! Shape type for MLPL arrays.

/// An ordered list of dimension sizes describing an array's shape.
///
/// - Scalar: rank 0, empty dims, elem_count = 1
/// - Vector: rank 1
/// - Matrix: rank 2
/// - Zero-size dimensions are allowed (elem_count = 0)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a shape from a dimension list.
    #[must_use]
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Rank-0 (scalar) shape with no dimensions.
    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// Rank-1 shape with the given length.
    #[must_use]
    pub fn vector(len: usize) -> Self {
        Self { dims: vec![len] }
    }

    /// Number of dimensions (rank).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Borrow the dimension slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Total number of elements.
    ///
    /// Returns 1 for scalar (empty dims). Returns 0 if any
    /// dimension is zero.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Compute strides for row-major layout. Shared by `transpose`,
/// `reduce_axis`, and `argmax_axis` (moved here from the retired
/// single-function ops_strides module).
pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}
