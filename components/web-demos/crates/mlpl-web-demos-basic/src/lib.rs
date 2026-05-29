//! Basic / intro / language-model demos. Each module exposes
//! `pub const` `Demo` entries that the facade crate's aggregator
//! threads into the global `DEMOS` list.

pub mod basics;
pub mod lm;
pub mod udf;
