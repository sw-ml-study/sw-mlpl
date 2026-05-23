//! Saga 33 step 008: variant-by-variant translation from
//! `mlpl-models-tape::TapeError` into mlpl-eval's `EvalError`.
//! Lives in its own file so `error.rs` stays under the 4-fn
//! Module-Function-Count warn line.

use crate::error::EvalError;

impl From<mlpl_models_tape::TapeError> for EvalError {
    fn from(e: mlpl_models_tape::TapeError) -> Self {
        match e {
            mlpl_models_tape::TapeError::UndefinedVariable(s) => Self::UndefinedVariable(s),
            mlpl_models_tape::TapeError::Unsupported(s) => Self::Unsupported(s),
            mlpl_models_tape::TapeError::ShapeMismatch {
                op,
                expected,
                actual,
            } => Self::ShapeMismatch {
                op,
                expected,
                actual,
            },
            mlpl_models_tape::TapeError::ArrayError(e) => Self::ArrayError(e),
        }
    }
}
