//! Aggregated `PATHS` list of every theme-specific
//! `LearningPath` const. Kept separate from `types.rs`
//! so `lib.rs` stays a pure facade.

use crate::types::LearningPath;

pub const PATHS: &[LearningPath] = &[
    crate::history::PATH_A_CHRONOLOGICAL_HISTORY_OF_ML,
    crate::architectures::PATH_ARCHITECTURE_ZOO__FROM_PIXELS_TO_LANGUAGE,
    crate::skills::PATH_BUILD_A_TRANSFORMER_FROM_PRIMITIVES,
    crate::skills::PATH_DATA___EXPLORATION,
    crate::skills::PATH_DIMENSIONALITY_REDUCTION,
    crate::architectures::PATH_OPTIMIZERS___REGULARIZATION,
    crate::skills::PATH_REPL_TO_SCRIPT,
    crate::architectures::PATH_TRAINING_PARADIGMS,
    crate::visual::PATH_VISION_TRANSFORMERS_IN_MLPL,
    crate::visual::PATH_VISUAL__ML_BY_DIAGRAM,
    crate::skills::PATH_ZERO_TO_LLM,
];
