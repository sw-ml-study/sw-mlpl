//! Error types for mlpl-array.

/// Errors produced by array operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayError {
    /// Data length does not match shape element count.
    DataLengthMismatch {
        /// Expected length (from shape).
        expected: usize,
        /// Actual data length provided.
        got: usize,
    },
    /// Reshape target element count differs from source.
    ShapeMismatch {
        /// Source element count.
        source: usize,
        /// Target element count.
        target: usize,
    },
    /// Index component is out of bounds for its axis.
    IndexOutOfBounds {
        /// Which axis.
        axis: usize,
        /// The index that was given.
        index: usize,
        /// The size of that axis.
        size: usize,
    },
    /// Index has wrong number of components for the array rank.
    RankMismatch {
        /// Expected rank.
        expected: usize,
        /// Number of index components given.
        got: usize,
    },
    /// Attempted to index an empty array.
    EmptyArray,
    /// Label list does not match array rank (Saga 11.5 Phase 2).
    LabelsRankMismatch {
        /// Array rank (number of axes).
        rank: usize,
        /// Number of labels the caller provided.
        labels: usize,
    },
    /// Two labeled operands disagree on axis labels (Saga 11.5 Phase 3).
    ///
    /// Raised by `apply_binop` when both sides carry labels but the
    /// label vectors differ. Step 006 will lift this into a
    /// richer `EvalError::ShapeMismatch` at the evaluator boundary.
    LabelMismatch {
        /// Labels on the first (left-hand) operand.
        expected: Vec<Option<String>>,
        /// Labels on the second (right-hand) operand.
        actual: Vec<Option<String>>,
    },
}

impl std::fmt::Display for ArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataLengthMismatch { expected, got } => {
                write!(
                    f,
                    "data length {got} does not match shape element count {expected}"
                )
            }
            Self::ShapeMismatch { source, target } => {
                write!(f, "shape mismatch: {source} vs {target} elements")
            }
            Self::IndexOutOfBounds { axis, index, size } => {
                write!(
                    f,
                    "index {index} out of bounds for axis {axis} with size {size}"
                )
            }
            Self::RankMismatch { expected, got } => {
                write!(
                    f,
                    "index has {got} components but array has rank {expected}"
                )
            }
            Self::EmptyArray => write!(f, "cannot index an empty array"),
            Self::LabelsRankMismatch { rank, labels } => {
                write!(
                    f,
                    "label list has {labels} entries but array has rank {rank}"
                )
            }
            Self::LabelMismatch { expected, actual } => {
                write!(
                    f,
                    "axis label mismatch: expected {expected:?}, got {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for ArrayError {}
