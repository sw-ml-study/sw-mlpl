//! Derived memory accounting for [`EngramSpec`]: the numbers the
//! REPL's `:describe` prints and the numbers a user must see
//! BEFORE a 12B-scale run (an accidental full-hidden-width table
//! is tens of gigabytes -- the doc's core cautionary tale).

use crate::spec::{EngramError, EngramSpec};

/// Storage type of the memory table, for byte accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
}

impl EngramSpec {
    /// Total rows of the ONE flattened memory table:
    /// `orders x heads x slots`.
    ///
    /// # Errors
    /// [`EngramError::Overflow`] when the product overflows.
    pub fn table_rows(&self) -> Result<usize, EngramError> {
        self.ngram_orders
            .len()
            .checked_mul(self.heads_per_ngram)
            .and_then(|v| v.checked_mul(self.slots_per_head))
            .ok_or(EngramError::Overflow)
    }

    /// Width of the concatenated retrieval per token:
    /// `orders x heads x head_dim`.
    ///
    /// # Errors
    /// [`EngramError::Overflow`] when the product overflows.
    pub fn retrieved_width(&self) -> Result<usize, EngramError> {
        self.ngram_orders
            .len()
            .checked_mul(self.heads_per_ngram)
            .and_then(|v| v.checked_mul(self.head_dim))
            .ok_or(EngramError::Overflow)
    }

    /// Memory-table parameters: `table_rows x head_dim`.
    ///
    /// # Errors
    /// [`EngramError::Overflow`] when the product overflows.
    pub fn parameter_count(&self) -> Result<u64, EngramError> {
        (self.table_rows()? as u64)
            .checked_mul(self.head_dim as u64)
            .ok_or(EngramError::Overflow)
    }

    /// Table bytes at `dtype`.
    ///
    /// # Errors
    /// [`EngramError::Overflow`] when the product overflows.
    pub fn bytes_for(&self, dtype: DType) -> Result<u64, EngramError> {
        let per_value = match dtype {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        };
        self.parameter_count()?
            .checked_mul(per_value)
            .ok_or(EngramError::Overflow)
    }
}
