//! Shared types + heads-up notes for the web playground demo
//! cluster. Extracted from mlpl-web during saga 82. Each themed
//! sub-crate (basic, vision, facade) defines its `Demo` consts
//! against this crate's `Demo` struct.

pub mod registry;

pub use registry::{Demo, PROGRESS_NOTES, ProgressNote, progress_notes_for};
