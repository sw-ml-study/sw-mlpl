//! Local error vocabulary for `mlpl-models-feasibility`.

use mlpl_array::ArrayError;
use mlpl_env_traits::DispatchError;

#[derive(Debug)]
pub enum FeasibilityError {
    BadArity {
        func: String,
        expected: usize,
        got: usize,
    },
    NotAModelName,
    UnknownModel(String),
    NotAScalar(String),
    NotPositive {
        func: String,
        name: String,
        value: f64,
    },
    BadEstimateShape(Vec<usize>),
    BadBudgetShape(Vec<usize>),
    NotAString(String),
    Dispatch(DispatchError),
    ArrayError(ArrayError),
}

impl From<ArrayError> for FeasibilityError {
    fn from(e: ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl From<DispatchError> for FeasibilityError {
    fn from(e: DispatchError) -> Self {
        Self::Dispatch(e)
    }
}

impl std::fmt::Display for FeasibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadArity {
                func,
                expected,
                got,
            } => write!(f, "{func} expects {expected} arguments, got {got}"),
            Self::NotAModelName => write!(
                f,
                "estimate_hypothetical: first argument must be a model-name string"
            ),
            Self::UnknownModel(name) => write!(
                f,
                "estimate_hypothetical: unknown model name '{name}' (try smollm-135m / smollm-360m / smollm-1.7b / llama-3.2-1b / qwen-2.5-0.5b)"
            ),
            Self::NotAScalar(func) => write!(f, "{func}: argument must be a scalar"),
            Self::NotPositive { func, name, value } => {
                write!(f, "{func}: {name} must be positive, got {value}")
            }
            Self::BadEstimateShape(dims) => {
                write!(f, "feasible: estimate must be rank-1 [5], got {dims:?}")
            }
            Self::BadBudgetShape(dims) => write!(
                f,
                "feasible: budget must be rank-1 [3] [vram, disk, wall], got {dims:?}"
            ),
            Self::NotAString(func) => {
                write!(f, "{func}: argument must be a string literal")
            }
            Self::Dispatch(e) => write!(f, "{e}"),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
        }
    }
}

impl std::error::Error for FeasibilityError {}
