//! Named axis dimension. `ShapeInfo` already carries a
//! `Vec<usize>` of dimension sizes; the IR adds optional axis
//! labels ("batch", "head", "token", "x", "y", ...) so
//! renderers can position controls (head selectors, batch
//! slicers) without re-deriving the layout from heuristics.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AxisDim {
    pub size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl AxisDim {
    #[must_use]
    pub fn unlabeled(size: usize) -> Self {
        Self { size, label: None }
    }

    #[must_use]
    pub fn labeled(size: usize, label: impl Into<String>) -> Self {
        Self {
            size,
            label: Some(label.into()),
        }
    }
}
