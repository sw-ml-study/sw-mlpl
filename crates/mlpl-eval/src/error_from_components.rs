//! Saga 33 step 011: variant-by-variant translation from
//! `components/mlpl-session/`'s sub-crate error types into
//! mlpl-eval's rich `EvalError`. Hosts every
//! `impl From<SubCrateError>` impl in one place so each
//! extraction adds an impl block (not a new module file).
//!
//! Loose-coupling boundary: each sub-crate exports its own
//! error vocabulary; this file's `From` impls map them onto
//! mlpl-eval's `EvalError` variants without the sub-crates
//! needing to know about `EvalError`.

use mlpl_models_freeze::FreezeError;
use mlpl_models_tape::TapeError;

use crate::error::EvalError;

impl From<TapeError> for EvalError {
    fn from(e: TapeError) -> Self {
        match e {
            TapeError::UndefinedVariable(s) => Self::UndefinedVariable(s),
            TapeError::Unsupported(s) => Self::Unsupported(s),
            TapeError::ShapeMismatch {
                op,
                expected,
                actual,
            } => Self::ShapeMismatch {
                op,
                expected,
                actual,
            },
            TapeError::ArrayError(e) => Self::ArrayError(e),
        }
    }
}

impl From<FreezeError> for EvalError {
    fn from(e: FreezeError) -> Self {
        match e {
            FreezeError::BadArity {
                func,
                expected,
                got,
            } => Self::BadArity {
                func,
                expected,
                got,
            },
            FreezeError::NotAModel { func, name } => {
                Self::Unsupported(format!("{func}: '{name}' is not a model"))
            }
        }
    }
}
