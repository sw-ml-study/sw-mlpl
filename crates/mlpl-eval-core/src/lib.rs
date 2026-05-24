//! Foundational eval-time types shared across mlpl-eval and
//! its sibling crates. Hosts the cleanest leaf modules from
//! the original mlpl-eval: `ModelSpec` + activation kind,
//! `TokenizerSpec`, the `MetricSink` trait, and the curated
//! builtin-groups table.
//!
//! Saga 33 step 019: added `TokenizerSpec` (moved from
//! mlpl-eval/src/tokenizer.rs) so downstream `Value`
//! extraction can proceed without a back-cycle through
//! mlpl-eval.

pub mod inspect_groups;
pub mod metric_sink;
pub mod model;
pub mod tokenizer;

pub use metric_sink::MetricSink;
pub use model::{ActKind, ModelSpec};
pub use tokenizer::TokenizerSpec;
