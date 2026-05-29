//! Web playground demo registry -- facade over the
//! mlpl-web-demos-* sub-crates. Re-exports the shared `Demo`
//! type, the progress-notes helpers, and the aggregated
//! `DEMOS` constant so mlpl-web can keep using `crate::demos::*`
//! call sites unchanged via a top-level `pub use` alias.

pub mod aggregator;
pub mod autoencoder;
pub mod dim_reduction;
pub mod gan;
pub mod models;
pub mod rnn;

pub use aggregator::DEMOS;
pub use mlpl_web_demos_types::{Demo, PROGRESS_NOTES, ProgressNote, progress_notes_for};
