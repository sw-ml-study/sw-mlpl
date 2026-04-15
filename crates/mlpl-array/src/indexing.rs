//! Indexing operations for DenseArray.

use crate::dense::DenseArray;
use crate::error::ArrayError;

impl DenseArray {
    /// Number of dimensions (delegates to shape).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// Total element count (delegates to shape).
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    /// Borrow the optional per-axis label slice.
    ///
    /// `None` means every axis is positional (the common case).
    /// `Some(slice)` has one entry per axis, with inner `None` for
    /// positional axes in a partially-labeled shape. Lives here
    /// rather than in `dense.rs` to group the short shape-metadata
    /// accessors (`rank`, `elem_count`, `labels`) together and keep
    /// `dense.rs` at its 7-function sw-checklist budget.
    #[must_use]
    pub fn labels(&self) -> Option<&[Option<String>]> {
        self.labels.as_deref()
    }

    /// Attach axis labels to this array, returning a new array that
    /// shares the original's shape and data but carries the given
    /// labels. Saga 11.5 Phase 2 entry point for the `label(x, [...])`
    /// built-in and the `x : [a, b]` annotation syntax.
    ///
    /// Returns `LabelsRankMismatch` if `labels.len()` differs from
    /// `self.rank()`. A rank-0 scalar accepts only the empty label
    /// list.
    pub fn with_labels(mut self, labels: Vec<Option<String>>) -> Result<Self, ArrayError> {
        if labels.len() != self.rank() {
            return Err(ArrayError::LabelsRankMismatch {
                rank: self.rank(),
                labels: labels.len(),
            });
        }
        self.labels = Some(labels);
        Ok(self)
    }

    /// Get a reference to the element at multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Result<&f64, ArrayError> {
        let offset = self.validate_index(index)?;
        Ok(&self.data()[offset])
    }

    /// Set the element at multi-dimensional index.
    pub fn set(&mut self, index: &[usize], value: f64) -> Result<(), ArrayError> {
        let offset = self.validate_index(index)?;
        self.data[offset] = value;
        Ok(())
    }

    /// Validate index and return flat offset.
    fn validate_index(&self, index: &[usize]) -> Result<usize, ArrayError> {
        if self.elem_count() == 0 {
            return Err(ArrayError::EmptyArray);
        }
        let dims = self.shape().dims();
        if index.len() != dims.len() {
            return Err(ArrayError::RankMismatch {
                expected: dims.len(),
                got: index.len(),
            });
        }
        let mut offset = 0;
        let mut stride = self.elem_count();
        for (axis, (&idx, &dim)) in index.iter().zip(dims.iter()).enumerate() {
            if idx >= dim {
                return Err(ArrayError::IndexOutOfBounds {
                    axis,
                    index: idx,
                    size: dim,
                });
            }
            stride /= dim;
            offset += idx * stride;
        }
        Ok(offset)
    }
}
