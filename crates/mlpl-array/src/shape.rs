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

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}
