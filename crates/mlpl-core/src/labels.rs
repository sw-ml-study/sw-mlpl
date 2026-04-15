//! Axis-label metadata for labeled-shape arrays (Saga 11.5, Phase 1).
//!
//! `LabeledShape` combines a dimension list with an optional per-axis
//! label. `None` entries mean "positional" so pre-existing code that
//! never touches labels keeps working unchanged. Later phases of
//! Saga 11.5 teach the existing ops (`transpose`, `matmul`,
//! `reduce_add`, ...) to propagate these labels and raise structured
//! `ShapeMismatch` errors when two labeled sides disagree.
//!
//! Lives in `mlpl-core` rather than `mlpl-array` because both
//! `mlpl-array` (DenseArray) and `mlpl-eval` (Value formatting,
//! error messages) will grow to depend on it, and `mlpl-core` is
//! the shared-types crate they both already reach into.

/// A dimension list with an optional per-axis string label.
///
/// - `dims` is the raw shape, same as `Shape::dims()`.
/// - `labels` has the same length as `dims`. A `None` entry means the
///   axis is positional (unlabeled); `Some(name)` marks it as labeled.
///
/// `LabeledShape::positional(dims)` is the default for any freshly
/// constructed array: the dim list is carried through but every axis
/// starts unlabeled. Labels only appear once a user explicitly calls
/// the Phase 2 `label(x, [...])` built-in or uses annotation syntax.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LabeledShape {
    dims: Vec<usize>,
    labels: Vec<Option<String>>,
}

impl LabeledShape {
    /// Create a labeled shape from dims and matching labels.
    ///
    /// Panics if `dims.len() != labels.len()`; the type's whole point
    /// is to keep the two sides in lock-step, so a mismatch is a bug
    /// in the caller rather than a recoverable runtime condition.
    #[must_use]
    pub fn new(dims: Vec<usize>, labels: Vec<Option<String>>) -> Self {
        assert_eq!(
            dims.len(),
            labels.len(),
            "LabeledShape: dims and labels must have the same length"
        );
        Self { dims, labels }
    }

    /// Create an all-positional labeled shape: every axis is `None`.
    #[must_use]
    pub fn positional(dims: Vec<usize>) -> Self {
        let labels = vec![None; dims.len()];
        Self { dims, labels }
    }

    /// Borrow the dimension list.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Borrow the per-axis label slice.
    #[must_use]
    pub fn labels(&self) -> &[Option<String>] {
        &self.labels
    }

    /// Does at least one axis carry an explicit label?
    #[must_use]
    pub fn is_labeled(&self) -> bool {
        self.labels.iter().any(Option::is_some)
    }
}

impl std::fmt::Display for LabeledShape {
    /// `[batch=2, feat=3]` for fully labeled, `[2, feat=3]` for
    /// partially labeled, `[2, 3]` for fully positional. Saga 11.5
    /// Phase 4: used in `EvalError::ShapeMismatch` rendering.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, (dim, label)) in self.dims.iter().zip(self.labels.iter()).enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match label {
                Some(name) => write!(f, "{name}={dim}")?,
                None => write!(f, "{dim}")?,
            }
        }
        f.write_str("]")
    }
}
