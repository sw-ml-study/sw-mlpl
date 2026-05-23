//! Local error vocabulary for the freeze/unfreeze operations.
//! Callers (e.g. mlpl-eval) translate via their own
//! `From<FreezeError>` impl.

#[derive(Debug, Clone, PartialEq)]
pub enum FreezeError {
    BadArity {
        func: String,
        expected: usize,
        got: usize,
    },
    NotAModel {
        func: String,
        name: String,
    },
}

impl std::fmt::Display for FreezeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadArity {
                func,
                expected,
                got,
            } => write!(f, "{func} expects {expected} arguments, got {got}"),
            Self::NotAModel { func, name } => write!(f, "{func}: '{name}' is not a model"),
        }
    }
}

impl std::error::Error for FreezeError {}
