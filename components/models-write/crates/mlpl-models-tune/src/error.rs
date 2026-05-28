//! Local error vocabulary for `mlpl-models-tune`.

use mlpl_array::ArrayError;
use mlpl_models_mutate::MutateError;

#[derive(Debug)]
pub enum TuneError {
    BadArity {
        func: String,
        expected: usize,
        got: usize,
    },
    NotAModel(String),
    NotAModelExpr,
    NotAScalar,
    BadRank(f64),
    ZeroRank,
    NestedLora,
    UndefinedVariable(String),
    NonRank2Linear {
        name: String,
        rank: usize,
    },
    RankTooLarge {
        rank: usize,
        in_dim: usize,
        out_dim: usize,
    },
    UnexpectedLoraInTree,
    Mutate(MutateError),
    ArrayError(ArrayError),
    RuntimeMessage(String),
}

impl From<ArrayError> for TuneError {
    fn from(e: ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl From<MutateError> for TuneError {
    fn from(e: MutateError) -> Self {
        Self::Mutate(e)
    }
}

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadArity {
                func,
                expected,
                got,
            } => write!(f, "{func} expects {expected} arguments, got {got}"),
            Self::NotAModel(name) => write!(f, "lora: '{name}' is not a model"),
            Self::NotAModelExpr => {
                write!(f, "lora: first argument must evaluate to a model")
            }
            Self::NotAScalar => {
                write!(f, "lora: rank, alpha, and seed must be scalars")
            }
            Self::BadRank(r) => write!(f, "lora: rank must be a non-negative integer, got {r}"),
            Self::ZeroRank => write!(f, "lora: rank must be positive, got 0"),
            Self::NestedLora => write!(
                f,
                "lora: model already has LoRA adapters; nested lora() is not supported"
            ),
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::NonRank2Linear { name, rank } => {
                write!(
                    f,
                    "lora: base Linear W '{name}' must be rank-2, got rank {rank}"
                )
            }
            Self::RankTooLarge {
                rank,
                in_dim,
                out_dim,
            } => write!(
                f,
                "lora: rank {rank} exceeds min(in={in_dim}, out={out_dim}) for this Linear"
            ),
            Self::UnexpectedLoraInTree => write!(
                f,
                "lora: unexpected LinearLora in source tree (nested lora check should have caught this)"
            ),
            Self::Mutate(e) => write!(f, "{e}"),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
            Self::RuntimeMessage(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for TuneError {}
