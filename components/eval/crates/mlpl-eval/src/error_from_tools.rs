//! `From<SubCrateError> for EvalError` impls for the tools /
//! workflow-domain sub-crates: tune, feasibility, llm,
//! loader-helpers. Saga 33 step 017 split.

use mlpl_loader_helpers::LoaderHelperError;
use mlpl_models_feasibility::FeasibilityError;
use mlpl_models_llm::LlmError;
use mlpl_models_tune::TuneError;

use crate::error::EvalError;

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

impl From<LlmError> for EvalError {
    fn from(e: LlmError) -> Self {
        match e {
            LlmError::BadArity { expected, got } => Self::BadArity {
                func: "llm_call".into(),
                expected,
                got,
            },
            LlmError::RuntimeMessage(msg) => Self::Unsupported(msg),
        }
    }
}

impl From<LoaderHelperError> for EvalError {
    fn from(e: LoaderHelperError) -> Self {
        match e {
            LoaderHelperError::ArrayError(arr) => Self::ArrayError(arr),
            other => Self::Unsupported(format!("{other}")),
        }
    }
}


#[cfg(feature = "image-io")]
impl From<mlpl_eval_image::ImageError> for crate::error::EvalError {
    fn from(e: mlpl_eval_image::ImageError) -> Self {
        crate::error::EvalError::Unsupported(e.0)
    }
}
