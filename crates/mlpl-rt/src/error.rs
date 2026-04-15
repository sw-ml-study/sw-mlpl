//! Error alias to keep mlpl-rt signatures compact.

/// Alias for the one fallible-result shape used across mlpl-rt.
pub type Result = std::result::Result<mlpl_array::DenseArray, mlpl_array::ArrayError>;
