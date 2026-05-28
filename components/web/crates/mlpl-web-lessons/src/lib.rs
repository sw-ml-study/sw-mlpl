//! Saga 32 step 003: tutorial-lesson static content for
//! apps/mlpl-web. Leaf crate.
//!
//! - `lessons`: core `Lesson` struct + the basic / intermediate
//!   lesson array `LESSONS`.
//! - `lessons_advanced`: saga-era deeper lessons (history,
//!   how-models-learn, why-backprop, lora, etc.) referenced by
//!   the LESSONS array via `crate::lessons_advanced::*`.
//! - `intro_md`: the tutorial-intro markdown-to-html renderer
//!   (`render_intro`).

pub mod intro_md;
pub mod lessons;
pub mod lessons_advanced;
pub mod lessons_dim_reduction;

pub use lessons::{LESSONS, Lesson};
