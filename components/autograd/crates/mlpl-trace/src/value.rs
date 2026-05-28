//! Trace value snapshots.

use serde::{Deserialize, Serialize};

/// A snapshot of a value at a point in evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TraceValue {
    /// A scalar value.
    Scalar { value: f64 },
    /// An array with shape and data.
    Array {
        /// Dimension sizes.
        shape: Vec<usize>,
        /// Flat row-major data.
        data: Vec<f64>,
        /// Per-axis labels (Saga 11.5 Phase 5). Omitted from JSON when
        /// the array is unlabeled, keeping the common-case trace terse.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        labels: Option<Vec<Option<String>>>,
    },
}

impl TraceValue {
    /// Create a TraceValue from a DenseArray.
    #[must_use]
    pub fn from_array(arr: &mlpl_array::DenseArray) -> Self {
        if arr.rank() == 0 {
            Self::Scalar {
                value: arr.data()[0],
            }
        } else {
            Self::Array {
                shape: arr.shape().dims().to_vec(),
                data: arr.data().to_vec(),
                labels: arr.labels().map(<[_]>::to_vec),
            }
        }
    }

    /// Convenience constructor for scalar.
    #[must_use]
    pub fn scalar(v: f64) -> Self {
        Self::Scalar { value: v }
    }
}
