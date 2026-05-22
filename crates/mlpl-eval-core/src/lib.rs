//! Foundational eval-time types shared across mlpl-eval and its
//! sibling crates (saga 32 step 001). Hosts the cleanest leaf
//! modules from the original mlpl-eval: `ModelSpec` + activation
//! kind, the `MetricSink` trait, and the curated builtin-groups
//! table.
//!
//! Future steps will pull `EvalError`, `Value`, and `Environment`
//! into this crate once their cross-deps (the `BreakSignal`
//! variant boxes a `Value`; `Value::Model` boxes a `ModelSpec`)
//! are untangled.

pub mod inspect_groups;
pub mod metric_sink;
pub mod model;

pub use metric_sink::MetricSink;
pub use model::{ActKind, ModelSpec};
