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

use mlpl_models_feasibility::FeasibilityError;
use mlpl_models_freeze::FreezeError;
use mlpl_models_inspect::InspectError;
use mlpl_models_mutate::MutateError;
use mlpl_models_tape::TapeError;
use mlpl_models_tune::TuneError;

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

impl From<MutateError> for EvalError {
    fn from(e: MutateError) -> Self {
        match e {
            MutateError::BadArity {
                func,
                expected,
                got,
            } => Self::BadArity {
                func,
                expected,
                got,
            },
            MutateError::NotAModel { func, name } => {
                Self::Unsupported(format!("{func}: '{name}' is not a model"))
            }
            MutateError::NotAModelExpr(func) => {
                Self::Unsupported(format!("{func}: argument must evaluate to a model"))
            }
            MutateError::UnknownFamily { family, valid } => Self::Unsupported(format!(
                "perturb_params: unknown family '{family}' (expected one of {})",
                valid.join(", ")
            )),
            MutateError::UndefinedVariable(name) => Self::UndefinedVariable(name),
            MutateError::ExpectedString(func) => Self::Unsupported(format!(
                "{func}: family (second argument) must be a string literal"
            )),
            MutateError::ExpectedScalar(func) => {
                Self::Unsupported(format!("{func}: expected a scalar"))
            }
            MutateError::ArrayError(e) => Self::ArrayError(e),
            MutateError::RuntimeMessage(msg) => Self::Unsupported(msg),
        }
    }
}

impl From<InspectError> for EvalError {
    fn from(e: InspectError) -> Self {
        match e {
            InspectError::BadArity {
                func,
                expected,
                got,
            } => Self::BadArity {
                func,
                expected,
                got,
            },
            InspectError::NotAModel { func, name } => {
                Self::Unsupported(format!("{func}: '{name}' is not a model"))
            }
            InspectError::NotAModelExpr(func) => {
                Self::Unsupported(format!("{func}: argument must evaluate to a model"))
            }
            InspectError::NoEmbedding => {
                Self::Unsupported("embed_table: model contains no Embedding layer".into())
            }
            InspectError::NoTrainableParams => {
                Self::Unsupported("estimate_train: model has no trainable parameters".into())
            }
            InspectError::NotAScalar { func, name, rank } => {
                Self::Unsupported(format!("{func}: {name} must be a scalar, got rank {rank}"))
            }
            InspectError::NotPositive { func, name, value } => {
                Self::Unsupported(format!("{func}: {name} must be positive, got {value}"))
            }
            InspectError::ArrayError(e) => Self::ArrayError(e),
        }
    }
}

impl From<TuneError> for EvalError {
    fn from(e: TuneError) -> Self {
        match e {
            TuneError::BadArity {
                func,
                expected,
                got,
            } => Self::BadArity {
                func,
                expected,
                got,
            },
            TuneError::NotAModel(name) => {
                Self::Unsupported(format!("lora: '{name}' is not a model"))
            }
            TuneError::NotAModelExpr => {
                Self::Unsupported("lora: first argument must evaluate to a model".into())
            }
            TuneError::NotAScalar => {
                Self::Unsupported("lora: rank, alpha, and seed must be scalars".into())
            }
            TuneError::BadRank(r) => Self::Unsupported(format!(
                "lora: rank must be a non-negative integer, got {r}"
            )),
            TuneError::ZeroRank => Self::Unsupported("lora: rank must be positive, got 0".into()),
            TuneError::NestedLora => Self::Unsupported(
                "lora: model already has LoRA adapters; nested lora() is not supported".into(),
            ),
            TuneError::UndefinedVariable(name) => Self::UndefinedVariable(name),
            TuneError::NonRank2Linear { name, rank } => Self::Unsupported(format!(
                "lora: base Linear W '{name}' must be rank-2, got rank {rank}"
            )),
            TuneError::RankTooLarge {
                rank,
                in_dim,
                out_dim,
            } => Self::Unsupported(format!(
                "lora: rank {rank} exceeds min(in={in_dim}, out={out_dim}) for this Linear"
            )),
            TuneError::UnexpectedLoraInTree => Self::Unsupported(
                "lora: unexpected LinearLora in source tree (nested lora check should have caught this)".into(),
            ),
            TuneError::Mutate(m) => Self::from(m),
            TuneError::ArrayError(e) => Self::ArrayError(e),
            TuneError::RuntimeMessage(msg) => Self::Unsupported(msg),
        }
    }
}

impl From<FeasibilityError> for EvalError {
    fn from(e: FeasibilityError) -> Self {
        match e {
            FeasibilityError::BadArity {
                func,
                expected,
                got,
            } => Self::BadArity {
                func,
                expected,
                got,
            },
            FeasibilityError::NotAModelName => Self::Unsupported(
                "estimate_hypothetical: first argument must be a model-name string".into(),
            ),
            FeasibilityError::UnknownModel(name) => Self::Unsupported(format!(
                "estimate_hypothetical: unknown model name '{name}' (try smollm-135m / smollm-360m / smollm-1.7b / llama-3.2-1b / qwen-2.5-0.5b)"
            )),
            FeasibilityError::NotAScalar(func) => {
                Self::Unsupported(format!("{func}: argument must be a scalar"))
            }
            FeasibilityError::NotPositive { func, name, value } => {
                Self::Unsupported(format!("{func}: {name} must be positive, got {value}"))
            }
            FeasibilityError::BadEstimateShape(dims) => Self::Unsupported(format!(
                "feasible: estimate must be rank-1 [5], got {dims:?}"
            )),
            FeasibilityError::BadBudgetShape(dims) => Self::Unsupported(format!(
                "feasible: budget must be rank-1 [3] [vram, disk, wall], got {dims:?}"
            )),
            FeasibilityError::NotAString(func) => {
                Self::Unsupported(format!("{func}: argument must be a string literal"))
            }
            FeasibilityError::Dispatch(d) => Self::Unsupported(format!("{d}")),
            FeasibilityError::ArrayError(e) => Self::ArrayError(e),
        }
    }
}
