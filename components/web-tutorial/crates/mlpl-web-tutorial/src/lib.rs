//! Tutorial panel for the web playground: the lesson index +
//! the run-an-example callback + the `TutorialView` enum that
//! switches between the index and a specific lesson body.
//! Extracted from mlpl-web during saga 82 so the mode-select
//! cluster (which needs `TutorialView` for the mode picker)
//! can carve out cleanly.
//!
//! mlpl-web re-exports `view::*` into both `crate::tutorial`
//! and `crate::components` so existing call sites stay
//! unchanged.

pub mod comment;
pub mod view;

pub use comment::split_inline_comment;
pub use view::{
    LESSONS, TutorialPanel, TutorialPanelProps, TutorialView, jump_lesson, run_example, step_lesson,
};
