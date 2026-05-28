//! Learning paths: curated ordered walks through the
//! tutorial / demo / diagram / glossary surfaces.
//!
//! A `LearningPath` is just a list of `Step`s, each of which
//! references existing content by name (lessons by title,
//! demos by name, diagrams by filename slug, glossary entries
//! by exact term). The walker view (`paths_view::PathsView`)
//! renders each step with a path-specific "why this is here"
//! framing and -- for lessons / demos -- a button that jumps
//! to the corresponding tab. Diagrams and glossary entries
//! render inline.
//!
//! Paths are pure data: adding a new path is one entry in
//! `PATHS` below, no UI changes needed.

#[derive(Clone, Copy, PartialEq)]
pub struct LearningPath {
    pub title: &'static str,
    pub blurb: &'static str,
    pub steps: &'static [Step],
}

#[derive(Clone, Copy, PartialEq)]
pub enum Step {
    /// A tutorial lesson, looked up by exact title.
    Lesson {
        title: &'static str,
        why: &'static str,
    },
    /// A demo, looked up by exact name.
    Demo {
        name: &'static str,
        why: &'static str,
    },
    /// A diagram, looked up by filename slug (matching the
    /// numbered `<slug>.svg` files in `diagrams/`).
    Diagram {
        slug: &'static str,
        why: &'static str,
    },
    /// A glossary entry, looked up by exact term (matching
    /// `## TermName` headers in `docs/glossary.md`).
    Glossary {
        term: &'static str,
        why: &'static str,
    },
    /// A path-orientation note that does not reference
    /// existing content. Shown as a small framing card.
    Note {
        title: &'static str,
        body: &'static str,
    },
}

pub const PATHS: &[LearningPath] = &[
    crate::paths_history::PATH_A_CHRONOLOGICAL_HISTORY_OF_ML,
    crate::paths_architectures::PATH_ARCHITECTURE_ZOO__FROM_PIXELS_TO_LANGUAGE,
    crate::paths_skills::PATH_BUILD_A_TRANSFORMER_FROM_PRIMITIVES,
    crate::paths_skills::PATH_DATA___EXPLORATION,
    crate::paths_skills::PATH_DIMENSIONALITY_REDUCTION,
    crate::paths_architectures::PATH_OPTIMIZERS___REGULARIZATION,
    crate::paths_skills::PATH_REPL_TO_SCRIPT,
    crate::paths_architectures::PATH_TRAINING_PARADIGMS,
    crate::paths_visual::PATH_VISION_TRANSFORMERS_IN_MLPL,
    crate::paths_visual::PATH_VISUAL__ML_BY_DIAGRAM,
    crate::paths_skills::PATH_ZERO_TO_LLM,
];
